# 1d_1w_adaptive_rsi_volatility_breakout
# Strategy: 1d RSI(14) with dynamic overbought/oversold levels based on volatility regime
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: In high volatility regimes, RSI extremes are more reliable for mean reversion.
# Uses ATR-based volatility to dynamically adjust RSI thresholds (more extreme in high vol).
# Filters trades with 1w EMA trend to avoid counter-trend moves in strong trends.
# Low frequency (~10-25/year) to minimize fee drag while capturing major reversals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_adaptive_rsi_volatility_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # ATR(14) for volatility measurement
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma = pd.Series(atr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma == 0, 1, atr_ma)  # Avoid division by zero
    
    # Dynamic RSI thresholds based on volatility
    # In high volatility, use more extreme levels (85/15); in low volatility, use standard (70/30)
    rsi_overbought = 70 + 15 * np.clip((atr_ratio - 1) / 2, 0, 1)  # Scales 70->85 as vol increases
    rsi_oversold = 30 - 15 * np.clip((atr_ratio - 1) / 2, 0, 1)   # Scales 30->15 as vol increases
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade in direction of weekly trend
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: RSI extremes with volatility-adjusted thresholds + trend alignment
        if (rsi_values[i] < rsi_oversold[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (rsi_values[i] > rsi_overbought[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: RSI returns to neutral zone (40-60) or trend reversal
        elif position == 1 and (rsi_values[i] > 50 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_values[i] < 50 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals