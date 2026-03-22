#!/usr/bin/env python3
"""
Experiment #490: 4h Volatility Spike Mean Reversion with Daily/Weekly HMA Regime

Hypothesis: After analyzing 479 failed experiments, the pattern is clear:
- Pure trend following fails in bear/range markets (2022 crash, 2025 sideways)
- Pure mean reversion fails in strong trends (2021 bull run)
- The KEY is VOLATILITY REGIME DETECTION + ASYMMETRIC logic

This strategy implements:

1. WEEKLY HMA(21) + DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Strong bull: price > 1w HMA AND price > 1d HMA (favor long only)
   - Strong bear: price < 1w HMA AND price < 1d HMA (favor short only)
   - Neutral: mixed signals (reduce position size by 50%)

2. VOLATILITY SPIKE DETECTION (ATR ratio):
   - ATR(7)/ATR(28) > 2.2 = panic/extreme vol (mean reversion likely)
   - ATR(7)/ATR(28) < 0.8 = vol compression (breakout likely)
   - This captures the "vol crush" pattern after panic selling

3. BOLLINGER BAND MEAN REVERSION:
   - Long: price < BB_lower(20, 2.5) + vol spike + bull/neutral regime
   - Short: price > BB_upper(20, 2.5) + vol spike + bear/neutral regime
   - Wider BB (2.5 std) ensures we only catch extreme moves

4. ADX REGIME FILTER WITH HYSTERESIS:
   - ADX > 28 = trending (disable mean reversion, enable breakout)
   - ADX < 18 = ranging (enable mean reversion)
   - Hysteresis prevents whipsaw at boundaries

5. CONNORS RSI FOR ENTRY TIMING:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long entry: CRSI < 15 (extreme oversold)
   - Short entry: CRSI > 85 (extreme overbought)
   - More responsive than standard RSI(14)

6. POSITION SIZING: 0.25 discrete (reduced from 0.30)
   - Neutral regime: 0.125 (half size)
   - Strong regime: 0.25 (full size)
   - Reduces exposure during uncertain periods

7. TRAILING STOP: 2.5 * ATR(14)
   - Tighter than previous 3.0*ATR for 4h timeframe
   - Signal → 0 when price moves 2.5*ATR against position

Why this should work on 4h:
- Volatility spike detection catches panic bottoms (2022 crash had multiple)
- Dual HTF (1d + 1w) provides robust trend filter without whipsaw
- Connors RSI is faster than RSI(14), generates more timely entries
- ADX hysteresis prevents rapid regime flips
- Should generate 25-40 trades/year per symbol (adequate for Sharpe)
- Asymmetric sizing protects during uncertain regimes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.125-0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_dual_htf_connors_bb_adx_hysteresis_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        positive_streaks = sum(1 for j in range(i-streak_period+1, i+1) if streak[j] > 0)
        if streak_period > 0:
            streak_rsi[i] = 100 * positive_streaks / streak_period
    
    # Percent Rank
    for i in range(pr_period, n):
        window = close[i-pr_period:i]
        count_lower = np.sum(window < close[i])
        pr = 100 * count_lower / pr_period
        crsi[i] = (rsi_short[i] + streak_rsi[i] + pr) / 3
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_vol_spike_ratio(high, low, close, short_period=7, long_period=28):
    """Calculate ATR ratio for volatility spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan)
    for i in range(len(close)):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    vol_ratio = calculate_vol_spike_ratio(high, low, close, 7, 28)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_FULL = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # ADX hysteresis state
    adx_trending = False  # True when ADX > 28, False when ADX < 18
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(adx[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === ADX REGIME WITH HYSTERESIS ===
        if adx[i] > 28:
            adx_trending = True
        elif adx[i] < 18:
            adx_trending = False
        
        trending_market = adx_trending
        ranging_market = not adx_trending
        
        # === DUAL HTF TREND BIAS ===
        above_1d = close[i] > hma_1d_aligned[i]
        above_1w = close[i] > hma_1w_aligned[i]
        
        # Strong bull: above both
        strong_bull = above_1d and above_1w
        # Strong bear: below both
        strong_bear = (not above_1d) and (not above_1w)
        # Neutral: mixed signals
        neutral_regime = not strong_bull and not strong_bear
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.2
        vol_compression = vol_ratio[i] < 0.8
        
        # === POSITION SIZING BASED ON REGIME ===
        if neutral_regime:
            SIZE = SIZE_HALF
        else:
            SIZE = SIZE_FULL
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only enter on volatility spike + extreme CRSI (mean reversion)
        if vol_spike and ranging_market:
            # Long: extreme oversold + bull/neutral regime
            if crsi[i] < 15 and not strong_bear:
                if close[i] < bb_lower[i]:
                    new_signal = SIZE
            
            # Short: extreme overbought + bear/neutral regime
            if crsi[i] > 85 and not strong_bull:
                if close[i] > bb_upper[i]:
                    new_signal = -SIZE
        
        # Breakout entries during vol compression + trending
        if vol_compression and trending_market:
            # Long breakout in bull regime
            if strong_bull and close[i] > bb_upper[i]:
                new_signal = SIZE_HALF
            
            # Short breakdown in bear regime
            if strong_bear and close[i] < bb_lower[i]:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if dual HTF trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and strong_bear:
                new_signal = 0.0
            if position_side < 0 and strong_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals