# Strategy: 12h_WilliamsAlligator_1dEMA200_TrendFilter_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.645 | -4.7% | -13.4% | 59 | DISCARD |
| ETHUSDT | 0.097 | +24.4% | -15.1% | 52 | KEEP |
| SOLUSDT | 1.050 | +144.2% | -21.3% | 51 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.109 | +7.0% | -9.3% | 20 | KEEP |
| SOLUSDT | 0.123 | +7.1% | -13.3% | 19 | KEEP |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA200 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and ATR-based volume confirmation.
- Williams Alligator: Jaw (EMA13 of Median Price, 8-period shift), Teeth (EMA8 of Median Price, 5-period shift), Lips (EMA5 of Median Price, 3-period shift).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA200 AND volume > 1.5 * 20-period average volume.
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA200 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Williams Alligator identifies trend absence, formation, and direction. Works in both trending and ranging markets.
- 1d EMA200 provides strong long-term trend filter to avoid counter-trend trades.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~80 total over 4 years (~20/year) based on Alligator crossover frequency with filters.
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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Williams Alligator components
    # Jaw: EMA13 of median price, 8-period shift
    jaw_raw = ema(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: EMA8 of median price, 5-period shift
    teeth_raw = ema(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: EMA5 of median price, 3-period shift
    lips_raw = ema(median_price, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_vol_ratio = volume[i] / (pd.Series(volume[max(0, i-19):i+1]).mean() + 1e-10)  # 12h volume ratio
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: bearish alignment (Lips < Teeth < Jaw) OR price falls below 1d EMA200
            if position == 1:
                if lips[i] < teeth[i] < jaw[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment (Lips > Teeth > Jaw) OR price rises above 1d EMA200
            elif position == -1:
                if lips[i] > teeth[i] > jaw[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend filter and volume confirmation
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] < jaw[i]
            
            # Long: Bullish alignment AND price > 1d EMA200 AND volume confirmation
            if bullish_alignment and curr_close > ema200_1d_aligned[i] and curr_vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < 1d EMA200 AND volume confirmation
            elif bearish_alignment and curr_close < ema200_1d_aligned[i] and curr_vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA200_TrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-24 21:58
