# Strategy: mtf_4h_volspike_fisher_donchian_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.039 | -5.4% | -18.3% | 899 | FAIL |
| ETHUSDT | -0.469 | +5.2% | -16.1% | 909 | FAIL |
| SOLUSDT | 0.139 | +26.8% | -13.2% | 897 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.155 | +7.5% | -5.8% | 273 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #499: 4h Primary + 1d HTF — Vol Spike Reversion + Fisher Transform + Donchian

Hypothesis: After 448 failed strategies (mostly CRSI/Choppiness/HMA combos), try a 
DIFFERENT approach based on proven research patterns:

1. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol → mean revert
   This captures "vol crush" after panic selloffs. Proven Sharpe 0.8-1.5 on BTC/ETH.
   
2. EHLERS FISHER TRANSFORM: period=9, catches reversals in bear rallies better than RSI
   Long when Fisher crosses above -1.5, short when crosses below +1.5
   
3. DONCHIAN BREAKOUT with 1d HMA trend filter: Only breakout in trend direction
   Prevents false breakouts against major trend
   
4. ASYMMETRIC REGIME: Different logic for bull vs bear (1d HMA slope)
   Bull: prefer long pullbacks. Bear: prefer short bounces + long only at extremes

Why this might beat current best (Sharpe=0.435):
- Vol spike reversion is DIFFERENT from CRSI/Choppiness (448 failed with those)
- Fisher Transform is proven for bear market reversals (research note #3)
- Fewer conflicting filters = more trades (critical: need >=30/symbol on train)
- 4h TF targets 20-50 trades/year (lower fee drag than 1h/30m)
- ATR 2.5x trailing stop protects in 2022-style crashes

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_fisher_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s) / 2.0
    
    # Normalize to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, 1e-10)
    
    normalized = (typical - lowest) / range_hl
    normalized = np.clip(normalized * 2.0 - 1.0, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    fisher = fisher.shift(1)  # Signal line (previous bar)
    
    return fisher.values, normalized.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # Vol spike ratio: ATR(7) / ATR(30)
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Fisher Transform for reversal timing
    fisher, fisher_norm = calculate_fisher_transform(high, low, period=9)
    
    # Donchian Channel for breakout detection
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Bollinger Bands for mean reversion extremes
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    
    # RSI for additional confirmation
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher crossover
    prev_fisher = np.zeros(n)
    prev_fisher[1:] = fisher[:-1]
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === VOL SPIKE DETECTION (mean reversion setup) ===
        vol_spike = vol_ratio[i] > 2.0  # Panic/extreme volatility
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_cross_up = (fisher[i] > -1.5) and (prev_fisher[i] <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (prev_fisher[i] >= 1.5)
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1]
        donchian_breakout_down = close[i] < donchian_lower[i-1]
        
        # === BOLLINGER BAND EXTREMES (mean reversion) ===
        bb_extreme_low = close[i] < bb_lower[i]
        bb_extreme_high = close[i] > bb_upper[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30.0
        rsi_overbought = rsi_14[i] > 70.0
        rsi_extreme_low = rsi_14[i] < 20.0
        rsi_extreme_high = rsi_14[i] > 80.0
        
        # === ENTRY LOGIC — VOL SPIKE REVERSION + FISHER TIMING ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple confluence conditions for frequency)
        # Condition 1: Vol spike + Fisher reversal + below BB (panic bottom)
        if vol_spike and fisher_cross_up and bb_extreme_low:
            new_signal = LONG_SIZE
        # Condition 2: Bull regime + Fisher extreme low (pullback in uptrend)
        elif bull_regime and fisher_extreme_low:
            new_signal = LONG_SIZE
        # Condition 3: Bull regime + Donchian breakout (trend continuation)
        elif bull_regime and hma_slope_bull and donchian_breakout_up:
            new_signal = LONG_SIZE * 0.8
        # Condition 4: RSI extreme + Fisher cross up (oversold reversal)
        elif rsi_extreme_low and fisher_cross_up:
            new_signal = LONG_SIZE
        # Condition 5: Below BB + RSI oversold (mean reversion)
        elif bb_extreme_low and rsi_oversold:
            new_signal = LONG_SIZE * 0.7
        # Condition 6: Vol spike alone with extreme RSI (panic capitulation)
        elif vol_spike and rsi_extreme_low:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (mirror logic for bear market)
        if new_signal == 0.0:
            # Condition 1: Vol spike + Fisher reversal + above BB (panic top)
            if vol_spike and fisher_cross_down and bb_extreme_high:
                new_signal = -SHORT_SIZE
            # Condition 2: Bear regime + Fisher extreme high (bounce in downtrend)
            elif bear_regime and fisher_extreme_high:
                new_signal = -SHORT_SIZE
            # Condition 3: Bear regime + Donchian breakdown (trend continuation)
            elif bear_regime and hma_slope_bear and donchian_breakout_down:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 4: RSI extreme + Fisher cross down (overbought reversal)
            elif rsi_extreme_high and fisher_cross_down:
                new_signal = -SHORT_SIZE
            # Condition 5: Above BB + RSI overbought (mean reversion)
            elif bb_extreme_high and rsi_overbought:
                new_signal = -SHORT_SIZE * 0.7
            # Condition 6: Vol spike alone with extreme RSI (panic FOMO top)
            elif vol_spike and rsi_extreme_high:
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on Fisher extreme high or RSI overbought
        if in_position and position_side > 0:
            if fisher_extreme_high or rsi_overbought:
                new_signal = 0.0
            # Exit if regime flips bearish
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
        
        # Exit short on Fisher extreme low or RSI oversold
        if in_position and position_side < 0:
            if fisher_extreme_low or rsi_oversold:
                new_signal = 0.0
            # Exit if regime flips bullish
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-23 05:10
