# Strategy: 6h_1dEMA21_BBWidth_RSI_VolFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.275 | -0.2% | -22.5% | 211 | FAIL |
| ETHUSDT | 0.368 | +49.7% | -17.9% | 223 | PASS |
| SOLUSDT | 1.118 | +276.3% | -29.4% | 239 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.416 | +14.2% | -10.3% | 77 | PASS |
| SOLUSDT | 0.270 | +10.8% | -12.3% | 72 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d = pd.Series(df_1d['high']).rolling(14, min_periods=14).max().values - \
             pd.Series(df_1d['low']).rolling(14, min_periods=14).min().values
    atr_1d = pd.Series(atr_1d).rolling(14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Bollinger Band width for squeeze detection
    sma_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate 1d EMA(21) for trend filter
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(bb_width_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # ATR-based volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Bollinger Band squeeze detection: bandwidth < 5%
        bb_squeeze = bb_width_1d_aligned[i] < 0.05
        
        # Trend filter: price > 1d EMA21 for long, price < 1d EMA21 for short
        trend_filter_long = price > ema_21_1d_aligned[i]
        trend_filter_short = price < ema_21_1d_aligned[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        rsi_filter = (rsi_1d_aligned[i] > 30) & (rsi_1d_aligned[i] < 70)
        
        # Additional volatility filter: 1d ATR > 1.5% of price
        vol_filter_1d = atr_1d_aligned[i] / price > 0.015 if price > 0 else False
        
        if position == 0:
            # Long setup: price above 1d EMA21 + volatility filter + not in squeeze + momentum filter + 1d vol filter
            if (trend_filter_long and vol_filter and not bb_squeeze and rsi_filter and vol_filter_1d):
                position = 1
                signals[i] = position_size
            # Short setup: price below 1d EMA21 + volatility filter + not in squeeze + momentum filter + 1d vol filter
            elif (trend_filter_short and vol_filter and not bb_squeeze and rsi_filter and vol_filter_1d):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d EMA21
            if price < ema_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d EMA21
            if price > ema_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dEMA21_BBWidth_RSI_VolFilter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 05:52
