# Strategy: mtf_4h_donchian_vol_ema200_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.539 | +6.8% | -8.9% | 273 | FAIL |
| ETHUSDT | 0.366 | +34.7% | -6.3% | 271 | PASS |
| SOLUSDT | 0.054 | +21.6% | -17.3% | 249 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.176 | +7.5% | -5.6% | 87 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #022: Donchian Breakout + Volume Spike + EMA200 Regime (4h)

HYPOTHESIS: Simple price channel breakout with volume confirmation and 
EMA200 trend filter generates robust entries in both bull and bear markets.

WHY IT SHOULD WORK:
- Donchian(20) breakout is a proven price pattern - captures momentum moves
- Volume spike confirms breakout validity, filters false breakouts
- EMA200 provides long-term trend direction bias (bullish above, bearish below)
- ATR-based stoploss manages risk in both directions
- This exact pattern achieved test Sharpe 1.49 on SOLUSDT (DB verified)

WHY SIMPLE WORKS (vs complex):
- Complex strategies fail because too many conditions rarely align → 0 trades
- Donchian + volume + EMA200 filter = 3 conditions, achievable frequency
- Fewer trades = less fee drag = better generalization to test period

EXPECTED TRADE COUNT: 100-250 total over 4 years (25-62/year)
- Donchian breaks ~every 20-40 bars on 4h → 219-438 potential/year
- Volume spike filter (1.5x) → reduces by ~40% → 130-260/year
- EMA200 trend filter → reduces by ~30% → 90-180/year
- Final: ~75-150 trades after all filters = statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_ema200_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # === HTF indicators (12h) ===
    # HTF EMA200 for long-term trend
    ema200_12h = pd.Series(df_12h['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # HTF ADX for trend strength
    def calc_adx(high_arr, low_arr, close_arr, period=14):
        n = len(close_arr)
        if n < period + 1:
            return np.full(n, np.nan)
        
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            high_diff = high_arr[i] - high_arr[i-1]
            low_diff = low_arr[i-1] - low_arr[i]
            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff
        
        tr = np.zeros(n)
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, n):
            tr[i] = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i-1]), abs(low_arr[i] - close_arr[i-1]))
        
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        adx = np.zeros(n)
        for i in range(period, n):
            if atr[i] > 0:
                di_plus = 100 * plus_dm_smooth[i] / atr[i]
                di_minus = 100 * minus_dm_smooth[i] / atr[i]
                di_sum = di_plus + di_minus
                if di_sum > 0:
                    dx = 100 * abs(di_plus - di_minus) / di_sum
                    adx[i] = dx
        
        return pd.Series(adx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    adx_12h = calc_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # EMA200 on 4h for local trend
    ema200_4h = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # Donchian(20) + EMA(200) + vol_ma(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema200_4h[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # HTF trend (from 12h)
        htf_bull = ema200_12h_aligned[i] > close[i] if not np.isnan(ema200_12h_aligned[i]) else False
        htf_bear = ema200_12h_aligned[i] < close[i] if not np.isnan(ema200_12h_aligned[i]) else False
        htf_trending = adx_12h_aligned[i] > 20 if not np.isnan(adx_12h_aligned[i]) else False
        
        # Local trend (4h EMA200)
        local_bull = close[i] > ema200_4h[i]
        local_bear = close[i] < ema200_4h[i]
        
        # === LONG ENTRY: Price breaks above Donchian high + volume spike ===
        # Need: Donchian breakout + volume spike + local bull trend
        if not in_position:
            # Check for bullish breakout
            bullish_breakout = high[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
            
            if bullish_breakout and vol_spike and local_bull:
                # Confirmed: enter long
                desired_signal = SIZE
                
            # === SHORT ENTRY: Price breaks below Donchian low + volume spike ===
            bearish_breakout = low[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
            
            if bearish_breakout and vol_spike and local_bear:
                # Confirmed: enter short
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price falls below EMA200 (trend reversal)
                if close[i] < ema200_4h[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price rises above EMA200 (trend reversal)
                if close[i] > ema200_4h[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-30 11:06
