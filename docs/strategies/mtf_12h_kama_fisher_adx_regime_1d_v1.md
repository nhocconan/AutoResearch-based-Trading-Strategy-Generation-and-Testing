# Strategy: mtf_12h_kama_fisher_adx_regime_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.530 | -11.5% | -18.2% | 216 | FAIL |
| ETHUSDT | 0.123 | +25.4% | -25.8% | 217 | PASS |
| SOLUSDT | 0.456 | +70.2% | -33.2% | 211 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.527 | +17.4% | -11.2% | 70 | PASS |
| SOLUSDT | 0.545 | +19.0% | -12.5% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #396: 12h Primary + 1d HTF — KAMA Trend + Fisher Transform + ADX Regime

Hypothesis: 12h timeframe with adaptive indicators beats static EMA/HMA approaches.
Key innovations vs #394:
1. KAMA (Kaufman Adaptive) instead of HMA - adapts to volatility, less whipsaw in 2022 crash
2. Ehlers Fisher Transform instead of CRSI - superior reversal detection in bear/range markets
3. ADX with HYSTERESIS (25 enter trend, 18 exit) - prevents rapid regime flipping
4. Asymmetric stops - tighter on shorts (2.0x ATR) vs longs (2.5x ATR) for bear market bias
5. Volatility-adjusted sizing - reduce position when ATR spikes > 2x median

Why this should beat Sharpe=0.612:
- KAMA proven in #392 notes (ETH Sharpe +0.755 with KAMA+ADX+Chop)
- Fisher Transform catches bear market reversals better than RSI/CRSI
- 12h TF = 20-40 trades/year = minimal fee drag (~1-2%)
- Different signal combination than #394 (KAMA+Fish+ADX vs HMA+CRSI+Chop+Donchian)

Target: Sharpe > 0.612, 25-45 trades/year, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_adx_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    Less lag in trends, more smoothing in chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    change = close_s.diff(er_period).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close_s.iloc[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for better reversal detection.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Normalize price within period range
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            continue
        
        # Normalize to -1 to +1
        normalized = 2.0 * (high[i] - lowest) / (highest - lowest) - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    atr = tr_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100.0 * plus_dm_s.ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100.0 * minus_dm_s.ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=21)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF KAMA for bias (1d)
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate median ATR for vol-adjusted sizing
    atr_median = np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (target 25-40 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX regime state (hysteresis)
    in_trend_regime = False  # Start neutral
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_21[i]) or np.isnan(kama_50[i]):
            continue
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trend regime when ADX > 25, exit when ADX < 18
        if adx[i] > 25.0:
            in_trend_regime = True
        elif adx[i] < 18.0:
            in_trend_regime = False
        
        # === HTF BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA crossover) ===
        kama_bullish = kama_21[i] > kama_50[i]
        kama_bearish = kama_21[i] < kama_50[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === VOL-ADJUSTED SIZING ===
        # Reduce size when ATR spikes > 2x median (high vol = reduce risk)
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.0:
            position_size = BASE_SIZE * 0.6  # Reduce to 60% in high vol
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.8  # Reduce to 80%
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP - Multiple confluence paths
        long_bias = price_above_kama_1d  # HTF bullish
        
        if long_bias:
            if in_trend_regime and kama_bullish:
                # Trend following long
                desired_signal = position_size
            elif not in_trend_regime and fisher_long:
                # Range mean-reversion long (Fisher reversal)
                desired_signal = position_size
            elif kama_bullish and fisher_long:
                # Pullback in uptrend with Fisher confirmation
                desired_signal = position_size
        
        # SHORT SETUP - Multiple confluence paths
        short_bias = price_below_kama_1d  # HTF bearish
        
        if short_bias:
            if in_trend_regime and kama_bearish:
                # Trend following short
                desired_signal = -position_size
            elif not in_trend_regime and fisher_short:
                # Range mean-reversion short (Fisher reversal)
                desired_signal = -position_size
            elif kama_bearish and fisher_short:
                # Rally in downtrend with Fisher confirmation
                desired_signal = -position_size
        
        # === STOPLOSS CHECK (Asymmetric: tighter on shorts) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr  # 2.5x for longs
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr  # 2.0x for shorts (tighter)
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === FISHER EXIT (reversal complete) ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            # Long exit when Fisher reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            # Short exit when Fisher reaches oversold
            desired_signal = 0.0
        
        # === TREND EXIT (HTF bias reversal) ===
        if in_position and position_side > 0 and price_below_kama_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_kama_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and long_bias:
                desired_signal = position_size
            elif position_side < 0 and short_bias:
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 09:45
