#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA trend filter and volume confirmation
# - Uses 4h EMA(34) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 1h Camarilla pivot levels (H3/L3) for breakout entries
# - Requires volume > 1.5 * 20-period volume average for confirmation
# - Fixed position size 0.20 to manage drawdown and reduce fee churn
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Works in bull markets via breakouts above resistance, in bear via breakdowns below support
# - Uses 4h for signal direction (HTF), 1h only for entry timing (LTF) to reduce trade frequency

name = "1h_4h_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08:00-20:00 UTC) - index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(34) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Pre-compute 1h ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Pre-compute 1h Camarilla pivot levels (using typical price)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3.0
    range_1h = prices['high'] - prices['low']
    
    camarilla_h3 = typical_price + (range_1h * 1.1 / 4)
    camarilla_l3 = typical_price - (range_1h * 1.1 / 4)
    camarilla_h4 = typical_price + (range_1h * 1.1 / 2)
    camarilla_l4 = typical_price - (range_1h * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if outside trading session or any required data is invalid
        if not in_session.iloc[i] or \
           np.isnan(ema_4h_aligned[i]) or np.isnan(atr[i]) or atr[i] <= 0 or \
           np.isnan(volume_confirm[i]) or np.isnan(camarilla_h3.iloc[i]) or \
           np.isnan(camarilla_l3.iloc[i]) or np.isnan(camarilla_h4.iloc[i]) or \
           np.isnan(camarilla_l4.iloc[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or mean reversion
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < camarilla_l3.iloc[i]:  # Mean reversion exit at L3
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: stoploss or mean reversion
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > camarilla_h3.iloc[i]:  # Mean reversion exit at H3
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for breakout entries in direction of 4h trend with volume confirmation
            if uptrend and close[i] > camarilla_h4.iloc[i] and volume_confirm[i]:  # Break above H4 in uptrend
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.20
            elif downtrend and close[i] < camarilla_l4.iloc[i] and volume_confirm[i]:  # Break below L4 in downtrend
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.20
    
    return signals