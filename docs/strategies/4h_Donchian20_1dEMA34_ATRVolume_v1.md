# Strategy: 4h_Donchian20_1dEMA34_ATRVolume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.014 | +20.6% | -12.0% | 176 | PASS |
| ETHUSDT | 0.137 | +26.7% | -18.7% | 173 | PASS |
| SOLUSDT | 0.547 | +73.3% | -17.6% | 173 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.983 | -2.7% | -7.7% | 62 | FAIL |
| ETHUSDT | 0.561 | +14.3% | -6.7% | 57 | PASS |
| SOLUSDT | 0.547 | +14.3% | -6.5% | 47 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volume confirmation
# Long when price breaks above Donchian upper band AND 1d close > 1d EMA34 (uptrend) AND volume > 1.5 * 20-bar ATR-scaled volume
# Short when price breaks below Donchian lower band AND 1d close < 1d EMA34 (downtrend) AND volume > 1.5 * 20-bar ATR-scaled volume
# Exit when price retraces to the Donchian midpoint (average of upper and lower bands)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d EMA34 provides strong trend filter for better regime adaptation in both bull and bear markets
# ATR-scaled volume threshold reduces false breakouts during low volatility periods

name = "4h_Donchian20_1dEMA34_ATRVolume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels for 4h timeframe (based on previous 20 bars)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR for volume confirmation (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume: volume > 1.5 * 20-bar average of (volume / ATR)
    # Avoid division by zero or near-zero ATR
    atr_safe = np.where(atr < 1e-10, np.nan, atr)
    volume_per_atr = volume / atr_safe
    avg_volume_per_atr_20 = pd.Series(volume_per_atr).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume_per_atr > (1.5 * avg_volume_per_atr_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper band AND uptrend AND volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND downtrend AND volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema34_1d_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to midpoint (mean reversion)
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to midpoint (mean reversion)
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-06 12:38
