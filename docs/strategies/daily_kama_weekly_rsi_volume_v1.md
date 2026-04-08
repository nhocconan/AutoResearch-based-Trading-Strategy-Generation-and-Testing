# Strategy: daily_kama_weekly_rsi_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.976 | -25.6% | -29.0% | 99 | FAIL |
| ETHUSDT | -0.091 | +9.5% | -31.5% | 97 | FAIL |
| SOLUSDT | 0.381 | +54.9% | -29.6% | 78 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.385 | +13.1% | -12.0% | 21 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend with weekly RSI filter and volume confirmation
# KAMA adapts to market noise, reducing whipsaws in choppy markets.
# Weekly RSI filters for extreme conditions in higher timeframe trend.
# Volume confirmation ensures institutional participation.
# Designed for low frequency in 1d timeframe (7-25 trades/year).
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "daily_kama_weekly_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = pd.Series(df_1w['close'].values)
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1w = (100 - (100 / (1 + rs))).values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily KAMA(10,2,30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = abs(close_s.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = sc.fillna(0)
    kama = [np.nan] * len(close)
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]) or np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend: price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Weekly RSI filter: avoid extremes
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price crosses below KAMA or RSI becomes overbought
            if not price_above_kama or rsi_1w_aligned[i] >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price crosses above KAMA or RSI becomes oversold
            if not price_below_kama or rsi_1w_aligned[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price above KAMA + RSI not overbought + volume confirmation
            if price_above_kama and rsi_not_overbought and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price below KAMA + RSI not oversold + volume confirmation
            elif price_below_kama and rsi_not_oversold and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 06:06
