# 1d_KAMA_RSI_Chop_Filter
# Hypothesis: On daily timeframe, KAMA adapts to market noise to identify trend direction.
# Combine with RSI for momentum and Choppiness Index to filter ranging vs trending regimes.
# Long when KAMA turns upward, RSI > 50, and market is trending (CHOP < 38.2).
# Short when KAMA turns downward, RSI < 50, and market is trending (CHOP < 38.2).
# Exit on opposite KAMA signal or when market becomes ranging (CHOP > 61.8).
# Designed for 1d to work in trending markets with ~15-25 trades per year.
# Uses 1-week trend filter to avoid counter-trend trades in stronger higher timeframe trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (adjusts to market noise)
    # Efficiency Ratio: price change over 10 periods / sum of absolute changes
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array alignment: volatility needs same length as change
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+10]))) for i in range(len(change))])
    # Pad beginning with NaN
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    sc = (er * 0.6 + 0.06) ** 2  # fast SC = 2/(2+1)=0.666, slow SC = 2/(30+1)=0.0645
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index (14)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (highest_high - lowest_low) > 0,
        100 * np.log10(sum_tr / atr / 14) / np.log10(14),
        50  # default when no range
    )
    
    # Get 1-week trend filter
    df_1w = get_htf_data(prices, '1w')
    # Weekly EMA34 for trend
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: current vs previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # Trend regime: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        trending = chop[i] < 38.2
        ranging = chop[i] > 61.8
        
        # Higher timeframe trend filter
        above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        below_weekly_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA up, RSI > 50, trending, and above weekly EMA
            if kama_up and rsi[i] > 50 and trending and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, trending, and below weekly EMA
            elif kama_down and rsi[i] < 50 and trending and below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR market becomes ranging
            if kama_down or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR market becomes ranging
            if kama_up or ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0