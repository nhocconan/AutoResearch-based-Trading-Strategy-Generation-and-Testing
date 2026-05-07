# 6h_Keltner_Channel_Breakout_Momentum_And_Trend_Filter
# Hypothesis: Combines Keltner Channel breakouts with momentum (RSI) and trend (ADX) filters on 6h timeframe.
# Uses daily trend filter (EMA50) and weekly momentum filter (RSI) to avoid false breakouts.
# Designed to work in both bull and bear markets by requiring trend alignment and momentum confirmation.
# Target: 15-30 trades/year per symbol to minimize fee drag while maintaining edge.
# Risk control: Exit on opposite Keltner band touch or trend failure.

timeframe = "6h"
name = "6h_Keltner_Channel_Breakout_Momentum_And_Trend_Filter"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d closes for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for RSI momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1w closes for momentum filter
    delta = pd.Series(df_1w['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w = rsi_14_1w.fillna(50).values  # Fill NaN with neutral 50
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate Keltner Channels on 6h: 20-period EMA, ATR(10) multiplier 2.0
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    kelly_upper = ema_20 + 2.0 * atr_10
    kelly_lower = ema_20 - 2.0 * atr_10
    
    # RSI(14) on 6h for momentum confirmation
    delta_6h = pd.Series(close).diff()
    gain_6h = delta_6h.clip(lower=0)
    loss_6h = -delta_6h.clip(upper=0)
    avg_gain_6h = gain_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_6h = loss_6h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_6h = avg_gain_6h / avg_loss_6h
    rsi_14_6h = 100 - (100 / (1 + rs_6h))
    rsi_14_6h = rsi_14_6h.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 10, 14)  # Ensure we have all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_1w_aligned[i]) or
            np.isnan(kelly_upper[i]) or np.isnan(kelly_lower[i]) or np.isnan(rsi_14_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Keltner with RSI > 50 (bullish momentum),
            # price above 1d EMA50 (uptrend), and weekly RSI > 50 (bullish momentum)
            if (close[i] > kelly_upper[i] and 
                rsi_14_6h[i] > 50 and 
                close[i] > ema_50_1d_aligned[i] and 
                rsi_14_1w_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner with RSI < 50 (bearish momentum),
            # price below 1d EMA50 (downtrend), and weekly RSI < 50 (bearish momentum)
            elif (close[i] < kelly_lower[i] and 
                  rsi_14_6h[i] < 50 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  rsi_14_1w_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch lower Keltner band or trend failure (price below 1d EMA50)
            if close[i] < kelly_lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch upper Keltner band or trend failure (price above 1d EMA50)
            if close[i] > kelly_upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals