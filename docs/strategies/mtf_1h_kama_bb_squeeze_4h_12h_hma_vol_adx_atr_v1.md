# Strategy: mtf_1h_kama_bb_squeeze_4h_12h_hma_vol_adx_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.978 | -22.6% | -26.8% | 4240 | FAIL |
| ETHUSDT | -0.287 | +11.6% | -7.9% | 282 | FAIL |
| SOLUSDT | 0.327 | +36.5% | -11.0% | 267 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.899 | +17.1% | -3.9% | 83 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #021: 1h KAMA Trend + 4h/12h HMA Filter + BB Squeeze Breakout

Hypothesis: After analyzing 20+ failed experiments, the pattern is clear:
1. Mean reversion (CRSI, RSI extremes) FAILS on 1h timeframe (Sharpe=-1.576 in #009)
2. Pure trend following gets whipsawed without volatility confirmation
3. The 12h KAMA+ADX+BB strategy (#017) showed promise: Sharpe=0.109, Return=+34%
4. Volume confirmation is UNDERUTILIZED in failed strategies

This 1h strategy combines:

1. 4h HMA + 12h HMA dual trend filter: Both must agree for direction.
   More robust than single HTF filter. Reduces false signals in chop.

2. KAMA (Kaufman Adaptive MA): Adapts to market efficiency ratio.
   Faster in trends, slower in ranges. Better than EMA for crypto.

3. Bollinger Band Squeeze: BB Width < 20th percentile = compression.
   Entry on breakout above/below BB with volume confirmation.

4. ADX Trend Strength: ADX(14) > 20 confirms trending conditions.
   Avoids entering breakouts in dead markets.

5. Volume Spike: Volume > 1.5x 20-bar average confirms breakout conviction.
   Critical filter missing from most failed strategies.

6. Asymmetric Sizing: 0.30 for strong signals (all filters agree), 0.20 for moderate.

7. ATR Trailing Stop: 2.5*ATR protects from reversals.

Why this should beat current best (Sharpe=0.123):
- KAMA adapts better than Supertrend in crypto's variable volatility
- Dual HTF filter (4h+12h) more robust than single 4h filter
- BB Squeeze + Volume captures explosive moves, not just trends
- Looser entry conditions than failed CRSI/RSI strategies = more trades

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year on 1h (optimal frequency per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_bb_squeeze_4h_12h_hma_vol_adx_atr_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER): |net change| / sum of absolute changes
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = np.abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        er[i] = net_change / sum_changes if sum_changes > 0 else 0
    
    # Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma * 100  # Percentage bandwidth
    
    return upper.values, lower.values, bandwidth.values, sma.values

def calculate_volume_spike(volume, lookback=20, threshold=1.5):
    """Detect volume spikes above threshold * average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=lookback, min_periods=lookback).mean()
    vol_spike = volume > (threshold * vol_avg)
    return vol_spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_width, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    vol_spike = calculate_volume_spike(volume, lookback=20, threshold=1.5)
    
    # Calculate BB Width percentile for squeeze detection
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x <= x.iloc[-1]) / len(x) * 100 if len(x) > 0 else 50
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.30  # All filters agree
    SIZE_MODERATE = 0.20  # Partial confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        if np.isnan(kama_21[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]):
            continue
        
        # === HTF TREND BIAS (4h + 12h HMA agreement) ===
        bull_htf = (close[i] > hma_4h_aligned[i]) and (close[i] > hma_12h_aligned[i])
        bear_htf = (close[i] < hma_4h_aligned[i]) and (close[i] < hma_12h_aligned[i])
        neutral_htf = not bull_htf and not bear_htf
        
        # === KAMA TREND (1h adaptive) ===
        kama_bull = close[i] > kama_21[i]
        kama_bear = close[i] < kama_21[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Trending market
        adx_weak = adx_14[i] < 20  # Range market
        
        # === BOLLINGER BAND SQUEEZE ===
        bb_squeeze = bb_width_percentile[i] < 20  # Bottom 20% = compression
        bb_breakout_long = close[i] > bb_upper[i]
        bb_breakout_short = close[i] < bb_lower[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        signal_strength = 0  # Count confirming filters
        
        # LONG ENTRY: HTF bull + KAMA bull + BB breakout + volume + ADX
        if bull_htf and kama_bull and di_bull:
            signal_strength += 1  # HTF trend
            signal_strength += 1  # KAMA trend
            signal_strength += 1  # DI direction
            
            if bb_breakout_long:
                signal_strength += 1  # BB breakout
                if vol_confirmed:
                    signal_strength += 1  # Volume confirmation
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            # Assign size based on confirmation count
            if signal_strength >= 5:
                new_signal = SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = SIZE_MODERATE
        
        # SHORT ENTRY: HTF bear + KAMA bear + BB breakout + volume + ADX
        elif bear_htf and kama_bear and di_bear:
            signal_strength += 1  # HTF trend
            signal_strength += 1  # KAMA trend
            signal_strength += 1  # DI direction
            
            if bb_breakout_short:
                signal_strength += 1  # BB breakout
                if vol_confirmed:
                    signal_strength += 1  # Volume confirmation
            
            if adx_strong:
                signal_strength += 1  # Trend strength
            
            # Assign size based on confirmation count
            if signal_strength >= 5:
                new_signal = -SIZE_STRONG
            elif signal_strength >= 3:
                new_signal = -SIZE_MODERATE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend reverses against position
            if position_side > 0 and bear_htf:
                trend_exit = True
            if position_side < 0 and bull_htf:
                trend_exit = True
            
            # Exit if KAMA crosses against position
            if position_side > 0 and kama_bear:
                trend_exit = True
            if position_side < 0 and kama_bull:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 20:09
