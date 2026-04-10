#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + chop filter (CHOP > 61.8 = range) on 1d timeframe
# - Long when KAMA trending up AND RSI < 40 (oversold in range) AND choppy regime (CHOP > 61.8)
# - Short when KAMA trending down AND RSI > 60 (overbought in range) AND choppy regime (CHOP > 61.8)
# - Exit when RSI returns to 50 (mean reversion to midpoint)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# - KAMA adapts to market noise, reducing false signals in choppy markets
# - RSI extremes in choppy regimes provide high-probability mean reversion setups
# - Chop filter ensures we only trade in ranging markets where mean reversion works

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for regime context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute KAMA (adaptive moving average) - ER=10, Fast=2, Slow=30
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.diff(volatility, prepend=volatility[0])
    er = np.zeros_like(close)
    er[1:] = change[1:] / (volatility[1:] + 1e-10)
    
    # Smoothing constant
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1=up, -1=down, 0=flat
    kama_dir = np.zeros_like(close)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, np.where(kama[1:] < kama[:-1], -1, 0))
    
    # Pre-compute RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute Choppiness Index (CHOP) - measures ranging vs trending
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - sum of TR over 14 periods
    atr14 = np.zeros_like(tr)
    atr14[13] = np.sum(tr[1:14])
    for i in range(14, len(tr)):
        atr14[i] = atr14[i-1] - (atr14[i-1] / 14) + tr[i]
    
    # Highest high and lowest low over 14 periods
    hh = np.zeros_like(high)
    ll = np.zeros_like(low)
    hh[13] = np.max(high[1:14])
    ll[13] = np.min(low[1:14])
    for i in range(14, len(high)):
        hh[i] = max(hh[i-1], high[i])
        ll[i] = min(ll[i-1], low[i])
    
    # Chop calculation
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(atr14[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Chop regime: > 61.8 = ranging (mean revert), < 38.2 = trending
    chop_regime = chop > 61.8
    
    # Align HTF indicators to 1d timeframe (though we're already on 1d, this ensures proper alignment)
    kama_dir_aligned = align_htf_to_ltf(prices, df_1w, kama_dir)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1w, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(kama_dir_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: KAMA up AND RSI oversold (<40) AND choppy regime
            if (kama_dir_aligned[i] == 1 and 
                rsi[i] < 40 and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: KAMA down AND RSI overbought (>60) AND choppy regime
            elif (kama_dir_aligned[i] == -1 and 
                  rsi[i] > 60 and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to RSI=50 (mean reversion)
            # Exit when RSI returns to 50 (mean reversion to midpoint)
            exit_long = (position == 1 and rsi[i] >= 50)
            exit_short = (position == -1 and rsi[i] <= 50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals