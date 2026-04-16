# 6h_RSI_MultiTimeframe_Divergence
# RSI divergence on 6h with 1d trend filter and volume confirmation
# Works in bull/bear: Mean reversion in ranges, trend-following in strong trends
# Target: 20-40 trades/year (80-160 total over 4 years)
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
    volume = prices['volume'].values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d EMA200 for trend filter ===
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h RSI(14) for divergence detection ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 6h volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        if np.isnan(rsi[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_spike[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema200 = ema200_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === Exit conditions ===
        if position == 1:  # Long
            if rsi_val > 70 or price < ema200:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            if rsi_val < 30 or price > ema200:
                signals[i] = 0.0
                position = 0
                continue
        
        # === Entry conditions (only when flat) ===
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = False
            if i >= 3:
                if (close[i] < close[i-2] and 
                    rsi[i] > rsi[i-2] and
                    rsi[i] < 40 and  # Oversold but not extreme
                    price > ema200):  # Above long-term trend
                    bull_div = True
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = False
            if i >= 3:
                if (close[i] > close[i-2] and 
                    rsi[i] < rsi[i-2] and
                    rsi[i] > 60 and  # Overbought but not extreme
                    price < ema200):  # Below long-term trend
                    bear_div = True
            
            if bull_div and vol_spike:
                signals[i] = 0.25
                position = 1
                continue
            elif bear_div and vol_spike:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_RSI_MultiTimeframe_Divergence"
timeframe = "6h"
leverage = 1.0