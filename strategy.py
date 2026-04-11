#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Long: price breaks above Camarilla H3 level, volume > 1.5x 20-period avg, 1w EMA(21) bullish (price > EMA)
# - Short: price breaks below Camarilla L3 level, volume > 1.5x 20-period avg, 1w EMA(21) bearish (price < EMA)
# - Exit: price returns to Camarilla pivot point (PP)
# - Uses discrete position sizing ±0.25 to limit drawdown and reduce fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits
# - Camarilla levels derived from prior day's range work well in both trending and ranging markets
# - 1w EMA filter ensures we only trade with the higher timeframe trend, reducing whipsaw in bear markets

name = "1d_1w_camarilla_ema_volume_v2"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return signals
    
    # Pre-compute 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # PP = (H + L + C) / 3
    # H3 = C + ((H-L) * 1.1/4)
    # L3 = C - ((H-L) * 1.1/4)
    PP = (high_1d + low_1d + close_1d) / 3
    H3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    L3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (same timeframe, so identity alignment)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Pre-compute 1d volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(PP_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        
        # Camarilla levels
        PP_level = PP_aligned[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # 1w EMA trend filter
        ema_bullish = close_price > ema_21_1w_aligned[i]
        ema_bearish = close_price < ema_21_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price above Camarilla H3, volume confirmation, 1w EMA bullish
        if close_price > H3_level and vol_confirm and ema_bullish:
            enter_long = True
        
        # Short breakout: price below Camarilla L3, volume confirmation, 1w EMA bearish
        if close_price < L3_level and vol_confirm and ema_bearish:
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
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals