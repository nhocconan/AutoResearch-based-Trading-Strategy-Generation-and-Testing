#!/usr/bin/env python3
"""
Experiment #167: 1d Primary + 1w HTF — Dual Regime Strategy (Choppiness + Connors RSI + Donchian)

Hypothesis: Pure trend-following failed in #164 because 2025 is bear/range market.
Pure mean-reversion failed in #155-166 because it gets crushed in strong trends.

SOLUTION: Dual regime adaptive strategy that SWITCHES based on Choppiness Index:
1) CHOP(14) > 61.8 = RANGING → Use Connors RSI mean reversion (buy oversold, sell overbought)
2) CHOP(14) < 38.2 = TRENDING → Use Donchian breakout with HMA trend filter

Key innovations:
- 1w HMA(21) for macro bias (only trade WITH weekly trend)
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Entry thresholds: CRSI < 15 (long), CRSI > 85 (short) in chop regime
- Donchian(20) breakout + HMA(21) alignment in trend regime
- ATR(14) trailing stop at 2.5x for risk management
- Position size: 0.25 base, 0.30 with HTF confluence

Why this should work:
- Adapts to market regime (the #1 factor in crypto)
- 1d timeframe = 20-50 trades/year (low fee drag)
- 1w HTF filter prevents counter-trend trades in strong moves
- Conservative sizing (0.25-0.30) limits drawdown in 2022-style crashes

Target: Sharpe > 0.5 on ALL symbols, 20-50 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_donchian_1w_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down days
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        window = streak[max(0, i-streak_period+1):i+1]
        if len(window) >= streak_period:
            # Map streak to 0-100: high positive = 100, high negative = 0
            avg_streak = np.mean(window)
            streak_rsi[i] = 50.0 + (avg_streak * 10.0)  # Scale streak to RSI range
            streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank - percentile of today's return over last 100 days
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window_returns = returns.iloc[i-pr_period+1:i+1].dropna()
        if len(window_returns) > 0:
            today_return = returns.iloc[i]
            rank = (window_returns < today_return).sum() / len(window_returns)
            percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging/choppy market
    CHOP < 38.2 = trending market
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    choppiness = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0  # neutral
    
    return choppiness

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    choppiness = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    # Calculate 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION ===
        is_choppy = choppiness[i] > 61.8  # Range/mean-reversion regime
        is_trending = choppiness[i] < 38.2  # Trend-following regime
        
        # === HTF MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D TREND BIAS ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- CHOPPY REGIME: MEAN REVERSION with Connors RSI ---
        if is_choppy:
            # Long: CRSI < 15 (oversold) + price above weekly HMA (macro bullish bias)
            if crsi[i] < 15.0 and price_above_hma_1w:
                new_signal = POSITION_SIZE_MAX
            # Short: CRSI > 85 (overbought) + price below weekly HMA (macro bearish bias)
            elif crsi[i] > 85.0 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_MAX
            # Partial entries without HTF confluence
            elif crsi[i] < 10.0:
                new_signal = POSITION_SIZE_BASE
            elif crsi[i] > 90.0:
                new_signal = -POSITION_SIZE_BASE
        
        # --- TRENDING REGIME: Donchian Breakout with HMA Filter ---
        elif is_trending:
            # Long breakout: price breaks Donchian upper + above both HMA21 and HMA1w
            if close[i] > donchian_upper[i-1] and price_above_hma_21 and price_above_hma_1w:
                new_signal = POSITION_SIZE_MAX
            # Short breakout: price breaks Donchian lower + below both HMA21 and HMA1w
            elif close[i] < donchian_lower[i-1] and price_below_hma_21 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_MAX
            # Partial entries with only 1d HMA confirmation
            elif close[i] > donchian_upper[i-1] and price_above_hma_21:
                new_signal = POSITION_SIZE_BASE
            elif close[i] < donchian_lower[i-1] and price_below_hma_21:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought (< 80) or trend still valid
                if (is_choppy and crsi[i] < 80.0) or (is_trending and price_above_hma_21):
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold (> 20) or trend still valid
                if (is_choppy and crsi[i] > 20.0) or (is_trending and price_below_hma_21):
                    new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d HMA in trend regime
        if in_position and position_side > 0 and is_trending and price_below_hma_21:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d HMA in trend regime
        if in_position and position_side < 0 and is_trending and price_above_hma_21:
            new_signal = 0.0
        
        # Exit mean-reversion trades when CRSI mean-reverts
        if in_position and position_side > 0 and is_choppy and crsi[i] > 60.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and is_choppy and crsi[i] < 40.0:
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
                # Position flip
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