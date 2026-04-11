#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with volume confirmation and 4h/1d trend filters
# - Long: price breaks above Camarilla H3 level, volume > 1.3x 20-period avg, price > 4h EMA(50) and > 1d EMA(50) (bullish bias)
# - Short: price breaks below Camarilla L3 level, volume > 1.3x 20-period avg, price < 4h EMA(50) and < 1d EMA(50) (bearish bias)
# - Exit: price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Target: 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# - Camarilla levels derived from prior day's range work well in both trending and ranging markets
# - Uses 4h and 1d EMA(50) for multi-timeframe trend alignment to reduce false breakouts

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for EMA trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # PP = (H + L + C) / 3
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    PP = (high_1d + low_1d + close_1d) / 3
    R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], access hour via index
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        PP_level = PP_aligned[i]
        R3_level = R3_aligned[i]
        S3_level = S3_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Multi-timeframe trend bias: price > 4h EMA(50) and > 1d EMA(50) for long bias
        ema_bias_long = (close_price > ema_50_4h_aligned[i]) and (close_price > ema_50_1d_aligned[i])
        # Short bias: price < 4h EMA(50) and < 1d EMA(50)
        ema_bias_short = (close_price < ema_50_4h_aligned[i]) and (close_price < ema_50_1d_aligned[i])
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla R3, volume confirmation, bullish MTF bias
        if close_price > R3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price below Camarilla S3, volume confirmation, bearish MTF bias
        if close_price < S3_level and vol_confirm and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to pivot point
            exit_long = close_price <= PP_level
        elif position == -1:
            # Exit short if price returns to pivot point
            exit_short = close_price >= PP_level
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.20
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals