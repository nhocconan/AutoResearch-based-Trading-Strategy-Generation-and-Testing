# Strategy: 6h_ElderRay_BullBearPower_12hEMA50_TrendFilter_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.243 | +30.6% | -10.5% | 627 | KEEP |
| ETHUSDT | 0.159 | +27.6% | -19.2% | 600 | KEEP |
| SOLUSDT | 0.514 | +60.4% | -20.9% | 558 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.250 | -4.2% | -8.7% | 228 | DISCARD |
| ETHUSDT | 0.451 | +12.0% | -8.4% | 211 | KEEP |
| SOLUSDT | 0.107 | +7.0% | -9.0% | 212 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend filter and ATR-based volume confirmation.
- Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close).
- Entry: Long when Bull Power > 0 AND price > 12h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when Bear Power < 0 AND price < 12h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Elder Ray signal OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Elder Ray measures bull/bear strength behind price moves, effective in both trending and ranging markets.
- 12h EMA50 provides medium-term trend filter to avoid counter-trend trades.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Works in bull markets (buy strength in uptrend) and bear markets (sell weakness in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Elder Ray crossover frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h, additional_delay_bars=1)
    
    # Elder Ray components: Bull Power and Bear Power (using 13-period EMA of close)
    ema13_close = ema(close, 13)
    bull_power = high - ema13_close
    bear_power = low - ema13_close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_vol_ratio = volume[i] / (pd.Series(volume[max(0, i-19):i+1]).mean() + 1e-10)  # 6h volume ratio
        
        # Exit conditions: opposite Elder Ray signal OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: Bear Power >= 0 OR price falls below 12h EMA50
            if position == 1:
                if bear_power[i] >= 0 or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power <= 0 OR price rises above 12h EMA50
            elif position == -1:
                if bull_power[i] <= 0 or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray signal with trend filter and volume confirmation
        if position == 0:
            # Long: Bull Power > 0 AND price > 12h EMA50 AND volume confirmation
            if bull_power[i] > 0 and curr_close > ema50_12h_aligned[i] and curr_vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price < 12h EMA50 AND volume confirmation
            elif bear_power[i] < 0 and curr_close < ema50_12h_aligned[i] and curr_vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-24 21:58
