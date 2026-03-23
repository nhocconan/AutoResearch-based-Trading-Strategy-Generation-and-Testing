#!/usr/bin/env python3
"""
Experiment #358: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Lower TF (30m) strategies fail because:
1. Too many trades → fee drag (need <80 trades/year)
2. No HTF bias → whipsaw in strong trends
3. Simple RSI extremes trigger too often

This strategy uses:
1. 1d HMA(21) as MACRO BIAS (hard filter: only long if price > 1d HMA)
2. 4h Choppiness Index for regime (CHOP>55=range/mean-revert, CHOP<45=trend/follow)
3. 30m Connors RSI for entry timing (CRSI<15 long, CRSI>85 short)
4. Session filter (8-20 UTC) to avoid low liquidity whipsaws
5. Volume filter (>0.8x 20-bar avg) to confirm participation
6. ATR(14) 2.5x trailing stop for risk management
7. Fisher Transform confirmation for cleaner signals

KEY INSIGHT: Connors RSI (CRSI) has 75% win rate for mean reversion when combined
with HTF bias. Using 4h CHOP for regime + 1d HMA for bias ensures we only trade
in the right direction. 30m is ONLY for entry timing within HTF trend.

TARGET: 40-80 trades/year on 30m, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
POSITION SIZE: 0.20-0.25 (smaller for lower TF to reduce fee impact)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_fisher_4h1d_regime_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(Streak): Measures consecutive up/down days
    PercentRank: Where current price ranks vs last 100 periods
    
    Entry: CRSI < 10-15 (oversold), CRSI > 85-90 (overbought)
    """
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - count consecutive up/down periods
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (positive streak = bullish, negative = bearish)
    # Use absolute streak for RSI calculation
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    streak_gain_s = pd.Series(streak_gain)
    streak_loss_s = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rsi = 100.0 - (100.0 / (1.0 + avg_streak_gain / (avg_streak_loss + 1e-10)))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # Percent Rank - where does current close rank vs last N periods
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100,
        raw=False
    ).fillna(50.0).values
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price into a Gaussian normal distribution for cleaner reversal signals.
    """
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (typical - lowest) / (highest - lowest + 1e-10)
    
    normalized = np.clip(normalized, 0.001, 0.999)
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    fisher_s = pd.Series(fisher)
    fisher_prev = fisher_s.shift(1).values
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # HMA for trend on 30m
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 4h Choppiness for regime (aligned to 30m)
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.22  # 22% position size for 30m (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        if not in_session:
            # Outside session: only maintain existing position, don't enter new
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H BIAS (additional confirmation) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        is_choppy = chop_4h_aligned[i] > 55.0  # Range regime
        is_trending = chop_4h_aligned[i] < 45.0  # Trend regime
        # Neutral regime (45-55): reduce position or stay flat
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Check for extreme CRSI readings (mean reversion setup)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # Fisher confirmation
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        fisher_cross_up = fisher[i] > fisher_prev[i] and fisher_prev[i] < -0.8
        fisher_cross_down = fisher[i] < fisher_prev[i] and fisher_prev[i] > 0.8
        
        # 30m trend
        hma_bullish_30m = hma_16[i] > hma_48[i]
        hma_bearish_30m = hma_16[i] < hma_48[i]
        
        if is_choppy:
            # RANGE REGIME: Mean reversion with CRSI extremes
            # LONG: CRSI<15 + Fisher<-1.2 + price>1d HMA + volume OK
            # SHORT: CRSI>85 + Fisher>1.2 + price<1d HMA + volume OK
            
            if price_above_hma_1d and price_above_hma_4h and crsi_oversold and fisher_oversold and volume_ok:
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and price_below_hma_4h and crsi_overbought and fisher_overbought and volume_ok:
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: Follow HTF trend with 30m pullback entry
            # LONG: 1d/4h bullish + 30m HMA bullish + CRSI<40 (pullback) + Fisher cross up
            # SHORT: 1d/4h bearish + 30m HMA bearish + CRSI>60 (pullback) + Fisher cross down
            
            if price_above_hma_1d and price_above_hma_4h and hma_bullish_30m:
                if crsi[i] < 40 and fisher_cross_up and volume_ok:
                    desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and price_below_hma_4h and hma_bearish_30m:
                if crsi[i] > 60 and fisher_cross_down and volume_ok:
                    desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 70:
            # Long position: exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30:
            # Short position: exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === FISHER EXIT ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if bias and regime still valid
            if position_side > 0:
                if price_above_hma_1d and price_above_hma_4h:
                    if (is_choppy and crsi[i] < 70 and fisher[i] < 1.5) or \
                       (is_trending and hma_bullish_30m):
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_1d and price_below_hma_4h:
                    if (is_choppy and crsi[i] > 30 and fisher[i] > -1.5) or \
                       (is_trending and hma_bearish_30m):
                        desired_signal = -BASE_SIZE
        
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