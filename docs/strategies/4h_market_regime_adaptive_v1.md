# Strategy: 4h_market_regime_adaptive_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.536 | -7.8% | -17.5% | 397 | FAIL |
| ETHUSDT | -0.129 | +8.3% | -17.6% | 381 | FAIL |
| SOLUSDT | 0.786 | +125.1% | -28.4% | 335 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.825 | +21.9% | -9.3% | 110 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_market_regime_adaptive_v1
# Hypothesis: Adaptive strategy using regime detection (ADX) with dual logic - trend following in strong trends, mean reversion in chop.
# Uses 4h price action with 1h regime filter to reduce whipsaw. Target: 20-30 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_market_regime_adaptive_v1"
timeframe = "4h"
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
    
    # 1h regime filter (ADX) - load once before loop
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate ADX on 1h data
    tr1 = high_1h[1:] - low_1h[1:]
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    up_move = high_1h[1:] - high_1h[:-1]
    down_move = low_1h[:-1] - low_1h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = wilders_smooth(tr, 14)
    plus_dm_smoothed = wilders_smooth(plus_dm, 14)
    minus_dm_smoothed = wilders_smooth(minus_dm, 14)
    
    plus_di = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
    minus_di = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1h = wilders_smooth(dx, 14)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # 4h indicators
    # EMA20 for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # RSI14 for mean reversion signals
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[np.nan], gain])
    loss = np.concatenate([[np.nan], loss])
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(ema20[i]) or np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or np.isnan(adx_1h_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        adx_val = adx_1h_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 1:  # Long position
            # Exit conditions
            if is_trending:
                # In trend: exit on close below EMA20
                if close[i] < ema20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In range: exit on RSI overbought or mean reversion
                if rsi[i] > 70:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if is_trending:
                # In trend: exit on close above EMA20
                if close[i] > ema20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In range: exit on RSI oversold or mean reversion
                if rsi[i] < 30:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if is_trending:
                # Trend following: enter on breakouts with volume
                if close[i] > ema20[i] and volume_ok:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < ema20[i] and volume_ok:
                    position = -1
                    signals[i] = -0.25
            elif is_ranging:
                # Mean reversion: enter on RSI extremes with volume
                if rsi[i] < 30 and volume_ok:
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70 and volume_ok:
                    position = -1
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 15:42
