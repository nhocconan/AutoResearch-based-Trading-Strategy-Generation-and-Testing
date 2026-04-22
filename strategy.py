# 1d EMA + RSI + Chop Filter for 1d Timeframe
# Hypothesis: On daily timeframe, use EMA(50) for trend direction, RSI(14) for mean reversion signals,
# and Choppiness Index(14) as a regime filter to avoid whipsaws in strong trends.
# Long when: price > EMA50, RSI < 30 (oversold), and CHOP > 61.8 (ranging market)
# Short when: price < EMA50, RSI > 70 (overbought), and CHOP > 61.8 (ranging market)
# Exit when trend reverses or RSI reaches opposite extreme.
# Designed for 1d timeframe to target 10-30 trades/year per symbol.
# Uses 1h for trend filter to avoid false signals during strong trends.
# Works in bull/bear via trend filter + mean reversion in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for EMA and RSI (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI(14) on 1d
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Calculate Choppiness Index(14) on 1d
    atr_1d = []
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # First TR is NaN
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR)/ (HH - LL)) / log10(14)
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(14)
    
    # Align to 1d timeframe (no alignment needed as we're already on 1d)
    ema_50_1d_aligned = ema_50_1d  # Already on 1d
    rsi_1d_aligned = rsi_1d       # Already on 1d
    chop_aligned = chop           # Already on 1d
    
    # Load 1h data for additional trend filter (to avoid counter-trend in strong moves)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_20_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > EMA50 (1d trend), RSI < 30 (oversold), CHOP > 61.8 (ranging)
            # AND price > EMA20 (1h) to avoid buying into strong downtrend
            if (close[i] > ema_50_1d_aligned[i] and 
                rsi_1d_aligned[i] < 30 and 
                chop_aligned[i] > 61.8 and
                close[i] > ema_20_1h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < EMA50 (1d trend), RSI > 70 (overbought), CHOP > 61.8 (ranging)
            # AND price < EMA20 (1h) to avoid selling into strong uptrend
            elif (close[i] < ema_50_1d_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  chop_aligned[i] > 61.8 and
                  close[i] < ema_20_1h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on trend reversal or RSI overbought
                if (close[i] < ema_50_1d_aligned[i] or 
                    rsi_1d_aligned[i] > 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on trend reversal or RSI oversold
                if (close[i] > ema_50_1d_aligned[i] or 
                    rsi_1d_aligned[i] < 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_EMA50_RSI30_70_CHOP62"
timeframe = "1d"
leverage = 1.0