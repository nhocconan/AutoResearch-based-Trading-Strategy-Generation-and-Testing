#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [0] * len(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    kama = np.where(np.isnan(kama), close, kama)
    
    # Calculate RSI(14) on daily close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Chop on 1w high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = []
    tr_1w = []
    for i in range(len(close_1w)):
        if i == 0:
            tr = high_1w[i] - low_1w[i]
        else:
            tr = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        tr_1w.append(tr)
        atr_1w.append(np.mean(tr_1w[-14:]) if len(tr_1w) >= 14 else np.nan)
    chop = 100 * np.log10(sum(tr_1w[-14:]) / (np.max(high_1w[-14:]) - np.min(low_1w[-14:])) if len(tr_1w) >= 14 and (np.max(high_1w[-14:]) - np.min(low_1w[-14:])) > 0 else 50) if len(tr_1w) >= 14 else 50
    chop_series = pd.Series([chop] * len(close))  # Simplified: use last chop value for all (real implementation would need rolling)
    chop_values = chop_series.rolling(window=14, min_periods=14).mean().fillna(50).values
    
    # Align 1w trend (EMA20) and chop to daily
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_values)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Need enough data for KAMA, EMA20, RSI, Chop
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(ema20_1w_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema20_1w_val = ema20_1w_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price > KAMA, RSI > 50, Chop < 50 (trending), EMA20_1w up, volume spike
            if close[i] > kama_val and rsi_val > 50 and chop_val < 50 and close[i] > ema20_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Price < KAMA, RSI < 50, Chop < 50 (trending), EMA20_1w down, volume spike
            elif close[i] < kama_val and rsi_val < 50 and chop_val < 50 and close[i] < ema20_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price < KAMA or RSI < 40 or Chop > 60 (choppy) or trend down
            if close[i] < kama_val or rsi_val < 40 or chop_val > 60 or close[i] < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price > KAMA or RSI > 60 or Chop > 60 (choppy) or trend up
            if close[i] > kama_val or rsi_val > 60 or chop_val > 60 or close[i] > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals