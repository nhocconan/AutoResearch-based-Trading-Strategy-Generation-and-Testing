#!/usr/bin/env python3
"""
Experiment #357: 1d Primary + 1w HTF — Dual Regime with CRSI + Donchian

Hypothesis: Daily timeframe needs fewer filters but higher conviction signals.
Previous 1d strategies failed because:
1. Too many AND conditions prevented trades (0 trades = auto-reject)
2. Fisher Transform alone too noisy on daily
3. Choppiness thresholds too narrow

This strategy uses PROVEN patterns from literature:
1. 1w HMA(21) as MACRO BIAS (simple bull/bear filter)
2. 1d Choppiness Index for regime (CHOP>55=range, CHOP<45=trend, 45-55=neutral)
3. RANGE REGIME: Connors RSI <15 long, >85 short (proven 75% win rate)
4. TREND REGIME: Donchian(20) breakout + 1w bias confirmation
5. ATR(14) trailing stop at 3.0x for risk management
6. RELAXED entry thresholds to ensure 15-25 trades/year on 1d

KEY INSIGHT: On 1d, fewer trades with higher conviction beats many weak signals.
CRSI catches mean reversion in choppy markets, Donchian catches trends.
1w HMA prevents trading against macro trend.

TARGET: 15-25 trades/year on 1d, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_donchian_1w_hma_v1"
timeframe = "1d"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate at extremes.
    """
    close_s = pd.Series(close)
    
    # RSI(3) - short term momentum
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        up_streaks = np.sum(streak[max(0, i-streak_period+1):i+1] > 0)
        streak_rsi[i] = (up_streaks / streak_period) * 100.0
    streak_rsi[:streak_period] = 50.0
    
    # Percent Rank - where current price ranks in last N days
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i]) / rank_period * 100.0
        percent_rank[i] = rank
    percent_rank[:rank_period] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_short.values + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range, CHOP < 38.2 = trend
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return highest, lowest

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 1d (target 15-25 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need enough data for CRSI rank_period
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA - HARD FILTER) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # High choppiness = range regime
        is_trending = chop[i] < 45.0  # Low choppiness = trend regime
        # 45-55 = neutral, can trade either direction
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME: Connors RSI mean reversion
        if is_choppy:
            # Long: CRSI < 15 + price above 1w HMA (bullish macro)
            if crsi[i] < 15 and price_above_hma_1w:
                desired_signal = BASE_SIZE
            
            # Short: CRSI > 85 + price below 1w HMA (bearish macro)
            elif crsi[i] > 85 and price_below_hma_1w:
                desired_signal = -BASE_SIZE
            
            # Fallback: less extreme CRSI in neutral macro
            elif crsi[i] < 20 and not price_below_hma_1w:
                desired_signal = BASE_SIZE * 0.7
            
            elif crsi[i] > 80 and not price_above_hma_1w:
                desired_signal = -BASE_SIZE * 0.7
        
        # TREND REGIME: Donchian breakout
        elif is_trending:
            # Long breakout: price breaks Donchian high + 1w bullish
            if close[i] >= donchian_high[i-1] and price_above_hma_1w:
                desired_signal = BASE_SIZE
            
            # Short breakout: price breaks Donchian low + 1w bearish
            elif close[i] <= donchian_low[i-1] and price_below_hma_1w:
                desired_signal = -BASE_SIZE
        
        # NEUTRAL REGIME: Either signal with reduced size
        else:
            if crsi[i] < 18 and price_above_hma_1w:
                desired_signal = BASE_SIZE * 0.7
            elif crsi[i] > 82 and price_below_hma_1w:
                desired_signal = -BASE_SIZE * 0.7
            elif close[i] >= donchian_high[i-1] and price_above_hma_1w:
                desired_signal = BASE_SIZE * 0.7
            elif close[i] <= donchian_low[i-1] and price_below_hma_1w:
                desired_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
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
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if regime and bias still valid
            if position_side > 0:
                if price_above_hma_1w:
                    if (is_choppy and crsi[i] < 70) or (is_trending and close[i] > donchian_low[i]):
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_1w:
                    if (is_choppy and crsi[i] > 30) or (is_trending and close[i] < donchian_high[i]):
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