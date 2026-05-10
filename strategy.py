# 4H_1D_VWAP_Reversion_Bollinger_Band
# Hypothesis: Mean reversion in 4h timeframe using VWAP deviation and Bollinger Bands.
# Long when price is below VWAP - 1std and touches lower Bollinger Band with RSI oversold.
# Short when price is above VWAP + 1std and touches upper Bollinger Band with RSI overbought.
# Uses 1d trend filter to avoid counter-trend trades and volume confirmation for confirmation.
# Designed for both bull and bear markets by focusing on mean reversion with trend alignment.
# Target: 25-40 trades/year per symbol (100-160 total over 4 years).

name = "4H_1D_VWAP_Reversion_Bollinger_Band"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0)
    
    # Standard deviation of price from VWAP for Bollinger Bands
    price_dev = typical_price - vwap
    # Use 20-period rolling std dev of price deviation
    price_dev_series = pd.Series(price_dev)
    vwap_std = price_dev_series.rolling(window=20, min_periods=20).std().values
    
    # Bollinger Bands around VWAP
    vwap_upper = vwap + (vwap_std * 2)
    vwap_lower = vwap - (vwap_std * 2)
    
    # RSI for overbought/oversold confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.3)
    
    # Align 1d trend to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(vwap[i]) or np.isnan(vwap_upper[i]) or np.isnan(vwap_lower[i]) or \
           np.isnan(rsi[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price below VWAP - 1std, touches lower BB, RSI oversold, 1d uptrend, volume confirmation
            if (close[i] < vwap[i] - vwap_std[i] and 
                close[i] <= vwap_lower[i] and 
                rsi[i] < 30 and 
                trend_up_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price above VWAP + 1std, touches upper BB, RSI overbought, 1d downtrend, volume confirmation
            elif (close[i] > vwap[i] + vwap_std[i] and 
                  close[i] >= vwap_upper[i] and 
                  rsi[i] > 70 and 
                  not trend_up_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above VWAP or RSI overbought
            if close[i] > vwap[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below VWAP or RSI oversold
            if close[i] < vwap[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals