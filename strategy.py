#!/usr/bin/env python3
"""
Experiment #255: 1h Regime-Adaptive Strategy with Choppiness Index + Connors RSI + 4h HMA

Hypothesis: 1h timeframe needs regime detection to switch between mean-reversion (range)
and trend-following (trending) modes. Using Choppiness Index (CHOP) to detect regime,
Connors RSI for mean-reversion entries in ranges, and EMA pullback for trend entries.
4h HMA provides higher timeframe trend bias to avoid counter-trend trades.

Why this might work on 1h:
- 1h has enough bars for statistical significance but less noise than 15m/30m
- CHOP(14) > 61.8 = range market (mean revert), CHOP < 38.2 = trending (trend follow)
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven 75% win rate
- 4h HMA bias prevents entering against major trend
- Different logic per regime = adapts to market conditions
- Conservative sizing (0.25) + 2.5*ATR stoploss controls drawdown

Key improvements over failed experiments:
- #249 (1h Supertrend): Sharpe=-1.412 - pure trend following fails in ranges
- #243 (1h Regime Z-score): Sharpe=-2.143 - regime filter too strict
- This uses CHOP for regime (proven in literature) + Connors RSI (high win rate)
- Looser entry thresholds for more trades (must have >=10 trades)
- 4h HMA bias only (not 1d/1w) to avoid over-filtering

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_connors_rsi_4h_hma_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar (simple true range for CHOP formula)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop[:period] = 50.0
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long entry: CRSI < 10 (oversold)
    Short entry: CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        pos_streaks = 0
        neg_streaks = 0
        for j in range(i - streak_period + 1, i + 1):
            if streak[j] > 0:
                pos_streaks += streak[j]
            elif streak[j] < 0:
                neg_streaks += abs(streak[j])
        
        total = pos_streaks + neg_streaks
        if total > 0:
            streak_rsi[i] = 100 * pos_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    crsi[:rank_period] = 50.0
    return crsi

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = trend bias (only trade in direction of 4h trend)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2 - 61.8 = neutral (use trend following with caution)
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- MEAN REVERSION ENTRIES (Range regime + Connors RSI extremes) ---
        if is_range:
            # Long: CRSI < 15 (oversold) + 4h not strongly bearish
            if crsi[i] < 15:
                if not bear_trend_4h or (bear_trend_4h and close[i] < bb_lower[i]):
                    new_signal = SIZE_BASE
            
            # Short: CRSI > 85 (overbought) + 4h not strongly bullish
            if crsi[i] > 85:
                if not bull_trend_4h or (bull_trend_4h and close[i] > bb_upper[i]):
                    new_signal = -SIZE_BASE
        
        # --- TREND FOLLOWING ENTRIES (Trend regime + EMA pullback) ---
        if is_trend:
            # Long: 4h bullish + price pulls back to EMA21 + RSI not overbought
            if bull_trend_4h and ema_21[i] > ema_50[i]:
                if close[i] < ema_21[i] * 1.01 and close[i] > ema_21[i] * 0.99:
                    if rsi_14[i] < 60 and rsi_14[i] > 40:
                        new_signal = SIZE_BASE
            
            # Short: 4h bearish + price pulls back to EMA21 + RSI not oversold
            if bear_trend_4h and ema_21[i] < ema_50[i]:
                if close[i] > ema_21[i] * 0.99 and close[i] < ema_21[i] * 1.01:
                    if rsi_14[i] > 40 and rsi_14[i] < 60:
                        new_signal = -SIZE_BASE
        
        # --- NEUTRAL REGIME (38.2 - 61.8): Use stricter trend following ---
        if not is_range and not is_trend:
            # Only enter with strong 4h trend confirmation
            if bull_trend_4h and ema_21[i] > ema_50[i]:
                if close[i] > ema_21[i] and rsi_14[i] > 50 and rsi_14[i] < 70:
                    new_signal = SIZE_BASE
            
            if bear_trend_4h and ema_21[i] < ema_50[i]:
                if close[i] < ema_21[i] and rsi_14[i] < 50 and rsi_14[i] > 30:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals