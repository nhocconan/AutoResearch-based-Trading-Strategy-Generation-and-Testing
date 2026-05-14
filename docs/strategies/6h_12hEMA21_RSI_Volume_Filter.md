# Strategy: 6h_12hEMA21_RSI_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.959 | -14.4% | -21.4% | 210 | FAIL |
| ETHUSDT | 0.174 | +28.9% | -13.1% | 195 | PASS |
| SOLUSDT | 0.243 | +35.9% | -26.4% | 161 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.567 | +14.7% | -8.5% | 68 | PASS |
| SOLUSDT | -1.363 | -13.8% | -18.5% | 63 | FAIL |

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(21) for trend filter
    ema_21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 12h RSI(14) for momentum filter
    close_12h = pd.Series(df_12h['close'])
    delta_12h = close_12h.diff()
    gain_12h = delta_12h.where(delta_12h > 0, 0)
    loss_12h = -delta_12h.where(delta_12h < 0, 0)
    avg_gain_12h = gain_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_12h = loss_12h.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_12h = avg_gain_12h / avg_loss_12h
    rsi_12h = (100 - (100 / (1 + rs_12h))).values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 6h ATR(14) for volatility filter and position sizing
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 60-period average volume for confirmation (6h * 10 = 60h ~ 2.5 days)
    vol_avg = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # ATR-based volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg[i] * 1.5) if not np.isnan(vol_avg[i]) else False
        
        # Trend filter: price > 12h EMA21 for long, price < 12h EMA21 for short
        trend_filter_long = price > ema_21_12h_aligned[i]
        trend_filter_short = price < ema_21_12h_aligned[i]
        
        # Momentum filter: RSI between 40 and 60 to avoid extremes and chop
        rsi_filter = 40 <= rsi_12h_aligned[i] <= 60
        
        if position == 0:
            # Long setup: price above 12h EMA21 + volume confirmation + volatility filter + RSI filter
            if (trend_filter_long and vol_confirm and vol_filter and rsi_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below 12h EMA21 + volume confirmation + volatility filter + RSI filter
            elif (trend_filter_short and vol_confirm and vol_filter and rsi_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 12h EMA21 OR RSI > 70 (overbought)
            if price < ema_21_12h_aligned[i] or rsi_12h_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 12h EMA21 OR RSI < 30 (oversold)
            if price > ema_21_12h_aligned[i] or rsi_12h_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12hEMA21_RSI_Volume_Filter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 05:13
