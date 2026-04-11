#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Uses 4h EMA(50) for trend bias: long when price > EMA, short when price < EMA
# - Uses 1d Camarilla levels (H3/L3) from prior day for breakout entries
# - Requires volume > 1.5x 20-period average for confirmation
# - Exits when price returns to Camarilla pivot point (PP)
# - Session filter: only trade 08-20 UTC to avoid low-liquidity hours
# - Discrete position sizing: ±0.20 to limit drawdown and reduce fee churn
# - Target: 15-25 trades/year (60-100 total over 4 years) to stay within fee limits
# - 1h timeframe chosen for better entry timing while using higher TFs for signal direction

name = "1h_4h_1d_camarilla_breakout_v1"
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
    
    # Pre-compute session hours once (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Pre-compute 4h EMA(50) for trend bias
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d OHLC for Camarilla levels (using prior day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # PP = (H + L + C) / 3
    # H3 = C + ((H-L) * 1.1/2)
    # L3 = C - ((H-L) * 1.1/2)
    PP = (high_1d + low_1d + close_1d) / 3
    H3 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    L3 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe (using completed 1d bars only)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Pre-compute 1h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(PP_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_4h_aligned[i])):
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
        
        # 4h EMA trend bias
        ema_bias_long = close_price > ema_50_4h_aligned[i]
        ema_bias_short = close_price < ema_50_4h_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Camarilla H3, volume confirmation, long bias
        if close_price > H3_level and vol_confirm and ema_bias_long:
            enter_long = True
        
        # Short breakout: price breaks below Camarilla L3, volume confirmation, short bias
        if close_price < L3_level and vol_confirm and ema_bias_short:
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