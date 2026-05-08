#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour 4-hour trend + 6-hour RSI mean reversion with volume confirmation
# Long when 4h trend is up (EMA34 > EMA89), 6h RSI < 30, and volume spike
# Short when 4h trend is down (EMA34 < EMA89), 6h RSI > 70, and volume spike
# Uses 4h for trend direction (reduces whipsaw), 6s for entry timing
# Volume spike avoids low-liquidity false signals
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_RSI_MeanReversion_4hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 89:
        return np.zeros(n)
    
    # Calculate 4h EMAs for trend filter
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_4h = pd.Series(close_4h).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    ema89_4h_aligned = align_htf_to_ltf(prices, df_4h, ema89_4h)
    
    # Calculate 6h RSI for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(ema89_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_4h_val = ema34_4h_aligned[i]
        ema89_4h_val = ema89_4h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: 4h uptrend, RSI oversold, volume spike
            if ema34_4h_val > ema89_4h_val and rsi_val < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: 4h downtrend, RSI overbought, volume spike
            elif ema34_4h_val < ema89_4h_val and rsi_val > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 4h trend turns down or RSI overbought
            if ema34_4h_val < ema89_4h_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 4h trend turns up or RSI oversold
            if ema34_4h_val > ema89_4h_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals