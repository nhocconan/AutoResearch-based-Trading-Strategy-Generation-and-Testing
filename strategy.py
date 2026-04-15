#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean-reversion strategy using 4h Bollinger Bands and 1h RSI with volume confirmation
# Uses 4h Bollinger Bands (20, 2.0) for mean-reversion zones and 1h RSI(14) for oversold/overbought
# Volume filter ensures trades occur during high conviction periods
# Designed for low trade frequency (target 15-35/year) to avoid fee drag
# Works in ranging markets (mean reversion at BB extremes) and trending markets (avoid trades against trend)
# Uses discrete position sizing (0.20) to minimize churn

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h Bollinger Bands (20, 2.0)
    close_4h = df_4h['close'].values
    sma20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma20_4h + 2.0 * std20_4h
    lower_bb_4h = sma20_4h - 2.0 * std20_4h
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # 1h volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Get aligned indicators
        upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb_4h)[i]
        lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb_4h)[i]
        
        # Skip if not enough data
        if np.isnan(upper_bb_aligned) or np.isnan(lower_bb_aligned) or np.isnan(rsi_1h[i]) or np.isnan(volume_ma[i]):
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * volume_ma[i]
        
        # Long conditions: price touches lower BB (oversold) AND RSI < 30 (oversold momentum)
        if close[i] <= lower_bb_aligned and rsi_1h[i] < 30 and vol_confirm and position <= 0:
            position = 1
            signals[i] = position_size
        # Short conditions: price touches upper BB (overbought) AND RSI > 70 (overbought momentum)
        elif close[i] >= upper_bb_aligned and rsi_1h[i] > 70 and vol_confirm and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit: price returns to middle of BB or RSI returns to neutral zone
        elif position == 1 and (close[i] >= sma20_4h[i//16] if i >= 16 else close[i] or rsi_1h[i] > 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= sma20_4h[i//16] if i >= 16 else close[i] or rsi_1h[i] < 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4h_BB_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0