# 1d_Hybrid_Trend_Momentum
# Combines 1-week EMA trend with daily RSI momentum and volume confirmation
# Designed for low trade frequency (10-25/year) with strong trend capture
# Works in bull markets via trend following and bear markets via mean reversion in ranges

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Hybrid_Trend_Momentum"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    1d strategy: Long when price > weekly EMA(50) AND RSI(14) > 55 with volume confirmation
                Short when price < weekly EMA(50) AND RSI(14) < 45 with volume confirmation
    Exit when price crosses back through weekly EMA or RSI reaches extreme opposite
    Uses weekly trend filter to avoid counter-trend trades
    Target: 15-25 trades/year
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.3x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure sufficient warmup for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma20[i]
        
        if position == 0:
            # Long: Above weekly EMA, bullish RSI, volume confirmation
            if close[i] > ema50_1w_aligned[i] and rsi_values[i] > 55 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Below weekly EMA, bearish RSI, volume confirmation
            elif close[i] < ema50_1w_aligned[i] and rsi_values[i] < 45 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Below weekly EMA OR RSI overbought (>70)
            if close[i] < ema50_1w_aligned[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Above weekly EMA OR RSI oversold (<30)
            if close[i] > ema50_1w_aligned[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals