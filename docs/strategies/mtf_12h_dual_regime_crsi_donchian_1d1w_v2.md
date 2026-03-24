# Strategy: mtf_12h_dual_regime_crsi_donchian_1d1w_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.380 | +2.4% | -15.2% | 345 | FAIL |
| ETHUSDT | 0.037 | +20.7% | -13.0% | 338 | PASS |
| SOLUSDT | 0.898 | +125.1% | -27.4% | 330 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.176 | +1.7% | -11.1% | 122 | FAIL |
| SOLUSDT | 0.289 | +10.9% | -11.8% | 131 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #442: 12h Primary + 1d/1w HTF — Dual Regime with Enhanced Entry Confluence

Hypothesis: Building on #436 (Sharpe=0.143), add 1w HTF for stronger trend bias and
improve entry confluence to boost Sharpe toward 0.612 baseline. Key improvements:
1. Add 1w HMA as additional HTF bias layer (1d + 1w agreement = stronger signal)
2. Relax CRSI thresholds further (25/75) to ensure 30-50 trades/year
3. Add KAMA slope confirmation for trend entries
4. Improve regime thresholds (CHOP 50/60 instead of 45/55) for cleaner switches
5. Add volume confirmation on breakouts (taker_buy_volume ratio)
6. Better position sizing: 0.25 base, 0.30 on strong confluence, 0.15 on weak

Target: Sharpe > 0.612, 120-200 trades over 4-year train, DD < -35%
Timeframe: 12h (proven to work best for swing trading crypto)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d1w_v2"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    n = len(close)
    
    # RSI(3) component
    rsi = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        up_streak = 0
        down_streak = 0
        
        for j in range(i, max(i - streak_period - 5, 0), -1):
            if j == 0:
                break
            if close[j] > close[j-1]:
                up_streak += 1
                down_streak = 0
            elif close[j] < close[j-1]:
                down_streak += 1
                up_streak = 0
            else:
                break
        
        streak = up_streak if up_streak > 0 else -down_streak
        
        if streak > 0:
            streak_rsi[i] = 50.0 + (streak / (streak_period + 1)) * 50.0
        elif streak < 0:
            streak_rsi[i] = 50.0 - (abs(streak) / (streak_period + 1)) * 50.0
        else:
            streak_rsi[i] = 50.0
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank component
    pct_rank = np.full(n, np.nan)
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.insert(returns, 0, 0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        pct_rank[i] = 100.0 * np.sum(window < current) / rank_period
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pct_rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i-er_period])
        noise = np.nansum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    er = np.clip(er, 0, 1)
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.nan_to_num(sc, nan=slow_sc)
    
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope (rate of change)."""
    n = len(kama)
    slope = np.full(n, np.nan)
    for i in range(lookback, n):
        if kama[i] > 0 and kama[i-lookback] > 0:
            slope[i] = (kama[i] - kama[i-lookback]) / kama[i-lookback] * 100.0
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    kama = calculate_kama(close, er_period=10)
    kama_slope = calculate_kama_slope(kama, lookback=5)
    
    # Calculate taker buy ratio for volume confirmation
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Calculate and align HTF HMA for bias (1d and 1w)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% base position size for 12h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(kama[i]) or np.isnan(kama_slope[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 60.0  # Range market
        regime_trend = chop[i] < 50.0  # Trending market
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias when 1d and 1w agree
        htf_bullish = price_above_hma_1d and price_above_hma_1w
        htf_bearish = price_below_hma_1d and price_below_hma_1w
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === PRIMARY TREND (12h HMA + KAMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        kama_slope_positive = kama_slope[i] > 0.5  # KAMA sloping up
        kama_slope_negative = kama_slope[i] < -0.5  # KAMA sloping down
        
        # === CRSI SIGNALS (Mean Reversion) — RELAXED THRESHOLDS ===
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        crsi_extreme_oversold = crsi[i] < 20.0
        crsi_extreme_overbought = crsi[i] > 80.0
        
        # === DONCHIAN BREAKOUT (Trend Follow) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = taker_ratio[i] > 0.55  # More buying pressure
        volume_bearish = taker_ratio[i] < 0.45  # More selling pressure
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0  # Count confluence factors
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 60) — MEAN REVERSION ===
        if regime_chop:
            # Long: CRSI oversold + HTF not strongly bearish
            if crsi_oversold and not htf_bearish:
                signal_strength = 1
                if crsi_extreme_oversold:
                    signal_strength = 2
                if volume_bullish:
                    signal_strength += 1
                desired_signal = position_size * (0.8 + 0.2 * signal_strength / 3)
            
            # Short: CRSI overbought + HTF not strongly bullish
            if crsi_overbought and not htf_bullish:
                if desired_signal == 0:
                    signal_strength = 1
                    if crsi_extreme_overbought:
                        signal_strength = 2
                    if volume_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * (0.8 + 0.2 * signal_strength / 3)
        
        # === REGIME 2: TRENDING (CHOP < 50) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout OR HMA bullish + KAMA confirmation
            if donchian_breakout_long:
                signal_strength = 1
                if htf_bullish:
                    signal_strength += 1
                if volume_bullish:
                    signal_strength += 1
                if kama_slope_positive:
                    signal_strength += 1
                desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
            elif hma_bullish and price_above_kama and kama_slope_positive:
                if desired_signal == 0:
                    signal_strength = 1
                    if htf_bullish:
                        signal_strength += 1
                    if volume_bullish:
                        signal_strength += 1
                    desired_signal = position_size * 0.7 * (0.8 + 0.2 * signal_strength / 3)
            
            # Short: Donchian breakdown OR HMA bearish + KAMA confirmation
            if donchian_breakout_short:
                if desired_signal == 0:
                    signal_strength = 1
                    if htf_bearish:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    if kama_slope_negative:
                        signal_strength += 1
                    desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
            elif hma_bearish and price_below_kama and kama_slope_negative:
                if desired_signal == 0:
                    signal_strength = 1
                    if htf_bearish:
                        signal_strength += 1
                    if volume_bearish:
                        signal_strength += 1
                    desired_signal = -position_size * 0.7 * (0.8 + 0.2 * signal_strength / 3)
        
        # === REGIME 3: TRANSITION (50-60) — REDUCED SIZE ===
        else:
            # Only strong signals
            if crsi_extreme_oversold and not htf_bearish:
                desired_signal = position_size * 0.5
            elif crsi_extreme_overbought and not htf_bullish:
                desired_signal = -position_size * 0.5
            elif donchian_breakout_long and htf_bullish:
                desired_signal = position_size * 0.5
            elif donchian_breakout_short and htf_bearish:
                desired_signal = -position_size * 0.5
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_kama):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or price_below_kama):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.30:
                    desired_signal = 0.30
                elif desired_signal >= 0.20:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.30:
                    desired_signal = -0.30
                elif desired_signal <= -0.20:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
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
2026-03-23 10:32
