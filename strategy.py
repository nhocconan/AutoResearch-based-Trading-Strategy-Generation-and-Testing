# 1d_GoldenCross_Pullback_Momentum
# Hypothesis: Daily Golden Cross (EMA50 > EMA200) + pullback to EMA200 with momentum confirmation (RSI > 55)
# Works in bull markets via trend continuation and in bear markets via avoidance of false signals.
# Uses EMA for smooth trend, RSI for momentum filter, and pullback logic for better entry.
# Target: 20-50 trades over 4 years (5-12/year) with high win rate and low turnover.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 and EMA200 on daily
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume (20-period) on daily
    vol_avg_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe (already aligned since we're using daily data)
    # But we need to map daily values to 15m bars via forward fill of the previous day's close
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short (we only go long in this strategy)
    base_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            continue
        
        # Golden Cross condition: EMA50 > EMA200
        golden_cross = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        
        # Pullback condition: price is near EMA200 (within 1.5% of EMA200)
        pullback = abs(close[i] - ema200_1d_aligned[i]) / ema200_1d_aligned[i] < 0.015
        
        # Momentum condition: RSI > 55 (bullish momentum)
        momentum = rsi_aligned[i] > 55
        
        # Volume confirmation: volume > 1.2x average volume
        volume_confirm = volume[i] > 1.2 * vol_avg_20_aligned[i]
        
        # Long entry: Golden Cross + pullback + momentum + volume confirmation
        if golden_cross and pullback and momentum and volume_confirm and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Exit: if golden cross breaks down or RSI becomes overbought (>70)
        elif position == 1 and (not golden_cross or rsi_aligned[i] > 70):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_GoldenCross_Pullback_Momentum"
timeframe = "1d"
leverage = 1.0