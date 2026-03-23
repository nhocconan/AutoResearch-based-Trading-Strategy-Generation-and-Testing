#!/usr/bin/env python3
"""
Experiment #291: 4h Primary + 1d/1w HTF — Regime-Adaptive Dual Mode

Hypothesis: Pure trend or pure mean-reversion fails because crypto alternates between
trending and ranging regimes. This strategy ADAPTS based on Choppiness Index:
- CHOP > 61.8 (choppy/range): Connors RSI mean reversion (75% win rate proven)
- CHOP < 38.2 (trending): Donchian breakout + HMA trend following
- 1d HMA(21) for macro bias filter
- 1w HMA(50) for ultra-long-term trend context
- ATR(14) 2.5x trailing stoploss on all positions

KEY INSIGHT from failures (#284, #289): Over-filtering kills trades. This version:
- Uses regime to SELECT entry type, not to block entries
- CRSI triggers in ranges (frequent), Donchian triggers in trends (less frequent)
- Combined should hit 20-50 trades/year target on 4h
- Position size: 0.28 (conservative for 4h volatility)

TARGET: Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL), trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_crsi_donchian_1d1w_atr_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_series = pd.Series(tr)
    atr_sum = tr_series.rolling(window=period, min_periods=period).sum()
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return np.nan_to_num(chop, nan=50.0)

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long signal)
    CRSI > 90 = overbought (short signal)
    """
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # PercentRank component
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    crsi = (rsi_3 + streak_rsi + percent_rank.values) / 3.0
    return np.nan_to_num(crsi, nan=50.0)

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-long-term trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, 50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for 4h volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 61.8  # Range market
        is_trending = choppiness[i] < 38.2  # Trend market
        # Neutral zone: 38.2 - 61.8 (use either mode)
        
        # === MACRO BIAS FILTERS (HTF) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        
        # 4h trend direction
        hma_4h_bullish = hma_16[i] > hma_48[i]
        hma_4h_bearish = hma_16[i] < hma_48[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # --- MODE 1: CHOPPY/RANGE (Mean Reversion with CRSI) ---
        if is_choppy:
            # Long: CRSI < 15 (oversold) + price above 1w HMA (long-term bullish bias)
            if crsi[i] < 15.0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
            # Short: CRSI > 85 (overbought) + price below 1w HMA (long-term bearish bias)
            elif crsi[i] > 85.0 and not price_above_hma_1w:
                desired_signal = -POSITION_SIZE
        
        # --- MODE 2: TRENDING (Breakout with Donchian + HMA) ---
        elif is_trending:
            # Long: Price breaks Donchian upper + HMA bullish + 1d bias bullish
            if close[i] > donchian_upper[i] and hma_4h_bullish and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            # Short: Price breaks Donchian lower + HMA bearish + 1d bias bearish
            elif close[i] < donchian_lower[i] and hma_4h_bearish and not price_above_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # --- MODE 3: NEUTRAL ZONE (Hybrid - lighter filters) ---
        else:
            # Allow both CRSI and breakout signals with relaxed filters
            if crsi[i] < 20.0:
                desired_signal = POSITION_SIZE
            elif crsi[i] > 80.0:
                desired_signal = -POSITION_SIZE
            elif close[i] > donchian_upper[i] and hma_4h_bullish:
                desired_signal = POSITION_SIZE
            elif close[i] < donchian_lower[i] and hma_4h_bearish:
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT (4h HMA crossover against position) ===
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit on mean reversion) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC (maintain position if trend intact) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_4h_bullish:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and hma_4h_bearish:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals