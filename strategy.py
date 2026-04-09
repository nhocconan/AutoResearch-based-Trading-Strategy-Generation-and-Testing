#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d KAMA trend filter + 4h RSI mean reversion with volume spike
# - Uses 1d Kaufman Adaptive Moving Average (KAMA) for trend direction (trend-following in trending markets, mean-reversion in ranging)
# - 4h RSI(14) for mean-reversion entries: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend
# - Volume confirmation: current volume > 2.0x 20-period average to ensure participation
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: KAMA adapts to market regime, RSI extremes provide mean-reversion entries with volume confirmation
# - Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years)

name = "4h_1d_kama_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    # Smoothing Constant (SC) = [ER * (fastest - slowest) + slowest]^2
    # where fastest = 2/(2+1), slowest = 2/(30+1)
    # KAMA = prev_KAMA + SC * (close - prev_KAMA)
    
    # Calculate ER components
    change_10 = np.abs(np.diff(close_1d, 10))  # |close[t] - close[t-10]|
    # Pad beginning with NaN for rolling sum
    change_10_padded = np.concatenate([np.full(9, np.nan), change_10])
    
    volatility_10 = np.abs(np.diff(close_1d, 1))  # |close[t] - close[t-1]|
    volatility_sum_10 = pd.Series(volatility_10).rolling(window=10, min_periods=10).sum().values
    
    # ER = |net change| / sum volatility
    er = change_10_padded / volatility_sum_10
    er = np.nan_to_num(er, nan=0.0)  # Replace NaN with 0 (no trend)
    er = np.clip(er, 0, 1)  # Bound between 0 and 1
    
    # Smoothing constants
    fastest = 2.0 / (2 + 1)   # 0.6667
    slowest = 2.0 / (30 + 1)  # 0.0645
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]  # Initialize
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe (wait for completed 1d bar)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Pre-compute 4h RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])  # First element is NaN
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # Set first value to neutral
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit long: RSI > 50 (mean reversion complete) or trend turns bearish
            if rsi[i] > 50 or close[i] < kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short: RSI < 50 (mean reversion complete) or trend turns bullish
            if rsi[i] < 50 or close[i] > kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: RSI extreme with volume confirmation and trend filter
            if volume_confirmed:
                # Long entry: RSI < 30 (oversold) in uptrend (price > KAMA)
                if rsi[i] < 30 and close[i] > kama_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI > 70 (overbought) in downtrend (price < KAMA)
                elif rsi[i] > 70 and close[i] < kama_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals