# Strategy: mtf_4h_ichimoku_alligator_vol_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.130 | +17.7% | -8.5% | 190 | FAIL |
| ETHUSDT | -0.269 | +12.3% | -8.9% | 174 | FAIL |
| SOLUSDT | 0.387 | +43.0% | -12.1% | 162 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.255 | +8.4% | -5.6% | 56 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #021: Ichimoku Cloud + Williams Alligator + Volume Confirmation (4h)

HYPOTHESIS: Combine two powerful trend detection systems for robust 4h entries:
1. Ichimoku Cloud - multi-dimensional trend (cloud, T-K cross, momentum) 
2. Williams Alligator - trend direction + momentum confirmation
3. Volume spike - trade validation
4. ATR stoploss - risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price above cloud + Alligator lines spread wide + TK bullish cross = strong long
- Bear: Price below cloud + Alligator lines spread wide + TK bearish cross = strong short
- Range: Cloud thickness + Alligator convergence = no trade (reduced whipsaws)
- Both Ichimoku and Alligator have separate bull/bear logic built in

KEY INSIGHT from DB analysis: The best performers (Sharpe 1.3-1.8) use
multiple confirming indicators with tight but not too tight conditions.
Ichimoku is explicitly mentioned as untried, Alligator is "novel" - high potential.

TARGET: 100-250 total trades over 4 years (25-62/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_ichimoku_alligator_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ichimoku(high, low, close, period_fast=9, period_medium=26, period_slow=52):
    """
    Ichimoku Cloud calculation
    Returns: tenkan, kijun, senkou_a, senkou_b, chikou
    """
    n = len(close)
    
    # Tenkan-sen (Conversion Line): (9-period highest high + 9-period lowest low) / 2
    tenkan = np.zeros(n)
    for i in range(n):
        start = max(0, i - period_fast + 1)
        hh = np.max(high[start:i+1])
        ll = np.min(low[start:i+1])
        tenkan[i] = (hh + ll) / 2
    
    # Kijun-sen (Base Line): (26-period highest high + 26-period lowest low) / 2
    kijun = np.zeros(n)
    for i in range(n):
        start = max(0, i - period_medium + 1)
        hh = np.max(high[start:i+1])
        ll = np.min(low[start:i+1])
        kijun[i] = (hh + ll) / 2
    
    # Chikou Span (Lagging Span): current close shifted back 26 periods
    chikou = np.full(n, np.nan)
    for i in range(period_medium, n):
        chikou[i] = close[i]
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward 26 periods
    senkou_a = np.full(n, np.nan)
    for i in range(period_medium, n):
        senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period highest high + 52-period lowest low) / 2
    senkou_b = np.full(n, np.nan)
    for i in range(period_slow, n):
        start = i - period_slow + 1
        hh = np.max(high[start:i+1])
        ll = np.min(low[start:i+1])
        senkou_b[i] = (hh + ll) / 2
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """
    Williams Alligator: SMMA-based trend detection
    - Jaw (blue): 13-period SMMA, offset 8
    - Teeth (red): 8-period SMMA, offset 5
    - Lips (green): 5-period SMMA, offset 3
    """
    n = len(close)
    
    # SMMA function
    def smma(data, period):
        result = np.zeros(n)
        if period <= 0:
            return result
        # First value is simple MA
        result[period-1] = np.mean(data[:period])
        # SMMA for rest
        for i in range(period, n):
            result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result
    
    jaw = smma(high, jaw_period)
    teeth = smma(close, teeth_period)  # Alligator uses high for jaw, close for others
    lips = smma(low, lips_period)
    
    return jaw, teeth, lips

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - trend strength
    ADX > 25 = strong trend
    ADX < 20 = weak/range
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
                adx[i] = dx
    
    # ADX is smoothed DX
    adx_smooth = pd.Series(adx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx_smooth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === Ichimoku on 1d for HTF trend direction ===
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # HTF: price above cloud = bull, below = bear
    htf_price = df_1d['close'].values
    htf_bullish = (htf_price > senkou_a_1d) & (htf_price > senkou_b_1d)
    htf_bearish = (htf_price < senkou_a_1d) & (htf_price < senkou_b_1d)
    
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Ichimoku on 4h
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 300  # Ichimoku needs 52+26=78, Alligator needs 13, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(jaw[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === ICHIMOKU SIGNALS ===
        # Price position relative to cloud
        above_cloud = close[i] > senkou_a[i] and close[i] > senkou_b[i]
        below_cloud = close[i] < senkou_a[i] and close[i] < senkou_b[i]
        
        # TK Cross (Tenkan-Kijun crossover - momentum shift)
        tk_bullish_cross = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_bearish_cross = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Cloud color (future bias)
        cloud_thick = abs(senkou_a[i] - senkou_b[i]) > atr_14[i] * 0.5  # Thick cloud = strong trend
        
        # Chikou confirmation (price vs 26 bars ago)
        chikou_above = close[i] > chikou[i - 26] if i >= 26 and not np.isnan(chikou[i - 26]) else False
        chikou_below = close[i] < chikou[i - 26] if i >= 26 and not np.isnan(chikou[i - 26]) else False
        
        # === ALLIGATOR SIGNALS ===
        # Alligator awake (lines separated = trending)
        alligator_wide = (jaw[i] > teeth[i] > lips[i]) or (jaw[i] < teeth[i] < lips[i])
        
        # Alligator mouth open (strong trending)
        jaw_teeth_spread = abs(jaw[i] - teeth[i])
        teeth_lips_spread = abs(teeth[i] - lips[i])
        alligator_open = jaw_teeth_spread > atr_14[i] * 0.3 and teeth_lips_spread > atr_14[i] * 0.2
        
        # Price action relative to Alligator
        price_above_alligator = close[i] > jaw[i] and close[i] > teeth[i] and close[i] > lips[i]
        price_below_alligator = close[i] < jaw[i] and close[i] < teeth[i] and close[i] < lips[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx[i] > 22  # Relaxed from 25 to get more trades
        weak_trend = adx[i] < 18
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === HTF TREND ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else False
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish Ichimoku + Alligator confirmation + volume + HTF bull
            # Conditions: Above cloud OR bullish TK cross, price above Alligator, alligator wide/open, vol spike
            bull_ichimoku = above_cloud or (tk_bullish_cross and close[i] > senkou_a[i])
            bull_alligator = price_above_alligator and (alligator_wide or alligator_open)
            
            if bull_ichimoku and bull_alligator and vol_spike and (strong_trend or cloud_thick):
                if htf_bull or htf_bullish_aligned[i] > 0:  # Either HTF says bull or neutral
                    desired_signal = SIZE
            
            # SHORT: Bearish Ichimoku + Alligator confirmation + volume + HTF bear
            # Conditions: Below cloud OR bearish TK cross, price below Alligator, alligator wide/open, vol spike
            bear_ichimoku = below_cloud or (tk_bearish_cross and close[i] < senkou_a[i])
            bear_alligator = price_below_alligator and (alligator_wide or alligator_open)
            
            if bear_ichimoku and bear_alligator and vol_spike and (strong_trend or cloud_thick):
                if htf_bear or htf_bearish_aligned[i] > 0:  # Either HTF says bear or neutral
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if trend weakens (price falls below alligator)
                if close[i] < jaw[i] and close[i] < teeth[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if trend weakens (price rises above alligator)
                if close[i] > jaw[i] and close[i] > teeth[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-30 09:14
