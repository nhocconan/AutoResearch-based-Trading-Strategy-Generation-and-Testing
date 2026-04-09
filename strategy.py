#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# - Uses 1h timeframe for precise entry timing (reduces slippage vs 4h)
# - 4h Camarilla pivot levels provide structural support/resistance for breakout direction
# - 1d volume > 1.5 * 20-period average confirms institutional participation
# - 4h EMA(50) filter ensures trading with higher timeframe trend (avoids counter-trend whipsaws)
# - Discrete position sizing (0.20) minimizes fee churn
# - Session filter (08-20 UTC) avoids low-liquidity Asian session noise
# - Target: 15-25 trades/year per symbol (60-100 total over 4 years) to stay within fee limits

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours once (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h Camarilla pivot levels (based on previous 4h bar's OHLC)
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Calculate pivot point
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    range_hl_4h = prev_high_4h - prev_low_4h
    
    # Camarilla levels for breakout
    h4_4h = pivot_4h + range_hl_4h * 1.1 / 2
    h3_4h = pivot_4h + range_hl_4h * 1.1 / 4
    l3_4h = pivot_4h - range_hl_4h * 1.1 / 4
    l4_4h = pivot_4h - range_hl_4h * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    h4_4h_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    l4_4h_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume confirmation: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    # 1h ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if outside trading session or missing data
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(h4_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or
            np.isnan(l3_4h_aligned[i]) or np.isnan(l4_4h_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: ATR stoploss or mean reversion
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < l3_4h_aligned[i]:  # Mean reversion exit (break below L3)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: ATR stoploss or mean reversion
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > h3_4h_aligned[i]:  # Mean reversion exit (break above H3)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for breakout entries with volume confirmation and trend filter
            # Long: price > H4 AND above 4h EMA50 AND volume confirmation
            if (close[i] > h4_4h_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_confirm_1d_aligned[i]):
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.20
            # Short: price < L4 AND below 4h EMA50 AND volume confirmation
            elif (close[i] < l4_4h_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_confirm_1d_aligned[i]):
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.20
    
    return signals