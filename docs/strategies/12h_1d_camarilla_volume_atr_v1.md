# Strategy: 12h_1d_camarilla_volume_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.017 | +18.0% | -13.9% | 149 | DISCARD |
| ETHUSDT | 0.120 | +25.6% | -21.0% | 134 | KEEP |
| SOLUSDT | 0.225 | +35.0% | -27.0% | 120 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.153 | +7.8% | -10.2% | 48 | KEEP |
| SOLUSDT | -0.033 | +3.6% | -14.0% | 44 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout + 1d volume confirmation + ATR volatility filter
# - Camarilla levels from 1d: L3 (support) and H3 (resistance) act as intraday pivot points
# - Long when price breaks above H3 with volume > 1.5x 20-period average
# - Short when price breaks below L3 with volume > 1.5x 20-period average
# - ATR filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid low volatility chop
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
# - 1d HTF provides reliable pivot levels, 12h timeframe balances frequency and cost

name = "12h_1d_camarilla_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d Camarilla levels (H3, L3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for ATR
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Camarilla levels: based on previous day's range
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Pre-compute 12h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions
        breakout_long = price_close > camarilla_h3_aligned[i-1]  # Close above previous period's H3
        breakout_short = price_close < camarilla_l3_aligned[i-1]  # Close below previous period's L3
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # ATR filter: trade only when short-term ATR > 0.5 * long-term ATR (avoid low volatility)
        atr_filter = atr_14_aligned[i] > 0.5 * atr_50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla H3 breakout + volume confirmation + ATR filter
        if breakout_long and vol_confirm and atr_filter:
            enter_long = True
        
        # Short: Camarilla L3 breakdown + volume confirmation + ATR filter
        if breakout_short and vol_confirm and atr_filter:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or volatility collapse
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price breaks below L3 OR volatility collapses
            exit_long = (price_close < camarilla_l3_aligned[i-1]) or (not atr_filter)
        elif position == -1:
            # Exit short if price breaks above H3 OR volatility collapses
            exit_short = (price_close > camarilla_h3_aligned[i-1]) or (not atr_filter)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
```

## Last Updated
2026-04-11 04:14
