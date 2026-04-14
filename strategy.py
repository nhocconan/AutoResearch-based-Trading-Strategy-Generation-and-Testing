#!/usr/bin/env python3
"""
1d Bollinger Band Width + RSI + Volume Confirmation Strategy
Hypothesis: In ranging markets (low Bollinger Band Width), price tends to revert to the mean.
We use Bollinger Band Width percentile to detect ranging regimes, RSI for mean reversion signals,
and volume confirmation to filter false signals. Works in both bull and bear markets because
it adapts to volatility regimes rather than assuming trend direction.
"""
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
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + bb_std * bb_std_dev
    bb_lower = bb_middle - bb_std * bb_std_dev
    
    # Bollinger Band Width
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Bollinger Band Width percentile (50-period lookback) to detect ranging regimes
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).rank(pct=True).values
    
    # RSI (14)
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume moving average (20-period)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position_size = 0.25
    
    for i in range(50, n):
        # Regime filter: ranging market (low Bollinger Band Width)
        # Using 30th percentile as threshold for ranging markets
        if bb_width_percentile[i] > 0.30:
            continue
            
        # Mean reversion signals with volume confirmation
        # Long: price near lower band + RSI oversold + volume confirmation
        if (close[i] <= bb_lower[i] * 1.01 and  # Allow small tolerance
            rsi_values[i] < 30 and
            volume[i] > vol_ma[i] * 1.5):
            signals[i] = position_size
            
        # Short: price near upper band + RSI overbought + volume confirmation
        elif (close[i] >= bb_upper[i] * 0.99 and  # Allow small tolerance
              rsi_values[i] > 70 and
              volume[i] > vol_ma[i] * 1.5):
            signals[i] = -position_size
            
        # Exit: RSI returns to neutral zone (40-60)
        elif 40 <= rsi_values[i] <= 60:
            signals[i] = 0.0
    
    return signals

name = "1d_BBWidth_RSI_Volume_MeanReversion"
timeframe = "1d"
leverage = 1.0