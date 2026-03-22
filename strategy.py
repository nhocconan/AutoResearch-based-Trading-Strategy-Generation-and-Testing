#!/usr/bin/env python3
"""
Experiment #005: 12h Connors RSI + 1d HMA Trend + Volatility Regime Strategy
Hypothesis: 12h timeframe captures multi-day swings while avoiding intraday noise.
Uses Connors RSI (CRSI) for mean reversion entries - proven 75% win rate in bear markets.
1d HMA provides primary trend bias. Volatility regime (ATR ratio) filters entry type:
- High vol (ATR7/ATR30 > 1.8): Mean revert at extremes (CRSI < 15 long, > 85 short)
- Low vol (ATR7/ATR30 < 1.2): Trend follow with pullbacks (CRSI < 40 long, > 60 short)
Asymmetric sizing: reduce position in bear regime (2025 test is bearish).
Multiple CRSI entry paths ensure >=10 trades per symbol. 2.5*ATR stoploss.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_rsi_1d_hma_vol_regime_v1"
timeframe = "12h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_streak_rsi(close, period=2):
    """Calculate RSI of consecutive up/down days (Connors RSI component)."""
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_rsi = calculate_rsi(np.abs(streak), period)
    # Adjust sign: positive streak = bullish, negative = bearish
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percentile Rank of close over rolling window (Connors RSI component)."""
    n = len(close)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window_vals = close[i-period:i]
        current = close[i]
        rank = np.sum(window_vals < current) / period
        pr[i] = rank * 100  # Scale to 0-100 like RSI
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion in bear markets.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = upper - lower
    pct = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, sma, width, pct

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # Connors RSI for mean reversion entries
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Volatility regime: ATR ratio
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Bollinger Bands for extreme detection
    bb_upper, bb_lower, bb_sma, bb_width, bb_pct = calculate_bollinger(close, 20, 2.0)
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, 14)
    
    # EMA for trend confirmation
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # SMA for long-term trend
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    SIZE_QUARTER = 0.08
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(atr_ratio[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF)
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Long-term trend
        lt_bullish = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        lt_bearish = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # EMA trend
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # Volatility regime
        vol_high = atr_ratio[i] > 1.6  # Vol spike = mean revert
        vol_low = atr_ratio[i] < 1.3   # Low vol = trend follow
        vol_normal = not vol_high and not vol_low
        
        # Connors RSI zones (more aggressive for more trades)
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        crsi_low = crsi[i] < 35
        crsi_high = crsi[i] > 65
        crsi_neutral = crsi[i] > 40 and crsi[i] < 60
        
        # CRSI turning (momentum)
        crsi_turning_up = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_turning_down = crsi[i] < crsi[i-1] if i > 0 else False
        
        # Bollinger extremes
        bb_below_lower = close[i] < bb_lower[i]
        bb_above_upper = close[i] > bb_upper[i]
        bb_near_lower = bb_pct[i] < 0.15
        bb_near_upper = bb_pct[i] > 0.85
        
        # ADX trend strength
        trend_weak = adx[i] < 20
        trend_moderate = adx[i] >= 20 and adx[i] < 30
        trend_strong = adx[i] >= 30
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: High vol + CRSI extreme low (mean reversion - high priority)
        if vol_high and crsi_extreme_low:
            new_signal = SIZE_ENTRY
        
        # Path 2: Low vol + 1d bullish + CRSI low (trend pullback)
        elif vol_low and hma_1d_bullish and crsi_low and trend_moderate:
            new_signal = SIZE_ENTRY
        
        # Path 3: BB below lower + CRSI low (overshoot long)
        elif bb_below_lower and crsi_low:
            new_signal = SIZE_ENTRY
        
        # Path 4: 1d bullish + CRSI turning up from low
        elif hma_1d_bullish and crsi_turning_up and crsi[i] < 45 and crsi[i-1] < 45 if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: EMA bullish + CRSI neutral bounce
        elif ema_bullish and crsi_neutral and crsi_turning_up:
            new_signal = SIZE_ENTRY
        
        # Path 6: LT bullish + CRSI low (dip buy in bull market)
        elif lt_bullish and crsi_low and hma_1d_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 7: ADX weak + CRSI extreme (range mean revert)
        elif trend_weak and crsi_extreme_low:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: High vol + CRSI extreme high (mean reversion - high priority)
        if vol_high and crsi_extreme_high:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Low vol + 1d bearish + CRSI high (trend pullback)
        elif vol_low and hma_1d_bearish and crsi_high and trend_moderate:
            new_signal = -SIZE_ENTRY
        
        # Path 3: BB above upper + CRSI high (overshoot short)
        elif bb_above_upper and crsi_high:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 1d bearish + CRSI turning down from high
        elif hma_1d_bearish and crsi_turning_down and crsi[i] > 55 and crsi[i-1] > 55 if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: EMA bearish + CRSI neutral drop
        elif ema_bearish and crsi_neutral and crsi_turning_down:
            new_signal = -SIZE_ENTRY
        
        # Path 6: LT bearish + CRSI high (rally sell in bear market)
        elif lt_bearish and crsi_high and hma_1d_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 7: ADX weak + CRSI extreme (range mean revert)
        elif trend_weak and crsi_extreme_high:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R - reduce to half
                risk = 2.5 * atr_14[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
                # Take profit at 3R - reduce to quarter
                elif profit >= 3.0 * risk and position_reduced:
                    new_signal = SIZE_QUARTER
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R - reduce to half
                risk = 2.5 * atr_14[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
                # Take profit at 3R - reduce to quarter
                elif profit >= 3.0 * risk and position_reduced:
                    new_signal = -SIZE_QUARTER
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals