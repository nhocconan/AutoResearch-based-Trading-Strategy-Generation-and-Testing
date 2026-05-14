# Strategy: 6h_RSI14Weekly_ATR14Daily_VolumeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.434 | -3.5% | -19.8% | 62 | FAIL |
| ETHUSDT | 0.050 | +19.3% | -21.0% | 63 | PASS |
| SOLUSDT | 0.900 | +148.2% | -24.9% | 35 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.728 | +22.1% | -11.6% | 27 | PASS |
| SOLUSDT | 0.000 | +0.0% | 0.0% | 0 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get weekly data for 14-period RSI (trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate RSI14 on weekly data
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.zeros_like(avg_gain_1w), where=avg_loss_1w!=0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Align weekly RSI14 to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get daily data for 20-period ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR14 on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR14 to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h ATR for stop loss
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h_1[0] = high[0] - low[0]
    tr_6h_2[0] = np.abs(high[0] - close[0])
    tr_6h_3[0] = np.abs(low[0] - close[0])
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # need weekly RSI, daily ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: weekly RSI between 30 and 70 (avoid extremes)
        trend_filter = (rsi_1w_aligned[i] >= 30) and (rsi_1w_aligned[i] <= 70)
        
        if position == 0:
            # Long entry: price above 6h open + 0.5*ATR, with volume and trend filter
            if (close[i] > open_prices[i] + 0.5 * atr_6h[i] and 
                vol_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 6h open - 0.5*ATR, with volume and trend filter
            elif (close[i] < open_prices[i] - 0.5 * atr_6h[i] and 
                  vol_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 6h open or ATR-based stop
            if close[i] < open_prices[i] - 1.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 6h open or ATR-based stop
            if close[i] > open_prices[i] + 1.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Extract open prices for use in the function
open_prices = None  # Will be set from prices['open'].values inside function

# Fix: move open_prices extraction inside function
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 14-period RSI (trend filter)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate RSI14 on weekly data
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.zeros_like(avg_gain_1w), where=avg_loss_1w!=0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Align weekly RSI14 to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get daily data for 20-period ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR14 on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR14 to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h ATR for stop loss
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h_1[0] = high[0] - low[0]
    tr_6h_2[0] = np.abs(high[0] - close[0])
    tr_6h_3[0] = np.abs(low[0] - close[0])
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # need weekly RSI, daily ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: weekly RSI between 30 and 70 (avoid extremes)
        trend_filter = (rsi_1w_aligned[i] >= 30) and (rsi_1w_aligned[i] <= 70)
        
        if position == 0:
            # Long entry: price above 6h open + 0.5*ATR, with volume and trend filter
            if (close[i] > open_prices[i] + 0.5 * atr_6h[i] and 
                vol_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 6h open - 0.5*ATR, with volume and trend filter
            elif (close[i] < open_prices[i] - 0.5 * atr_6h[i] and 
                  vol_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 6h open or ATR-based stop
            if close[i] < open_prices[i] - 1.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 6h open or ATR-based stop
            if close[i] > open_prices[i] + 1.5 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI14Weekly_ATR14Daily_VolumeFilter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-18 11:23
