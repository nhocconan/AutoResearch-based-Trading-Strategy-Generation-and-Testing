#!/usr/bin/env python3
"""
Experiment #001: 4h Regime-Adaptive Strategy with 1d Trend Filter

Hypothesis: Single-regime strategies fail because BTC/ETH alternate between trending
and ranging markets. This strategy adapts based on Choppiness Index:
- CHOP > 61.8 (choppy): Mean reversion via Connors RSI extremes
- CHOP < 38.2 (trending): Trend following via HMA + Donchian breakout
- 1d HMA(21) provides major trend bias (only long if price > 1d HMA, vice versa)

Why 4h works:
- Natural 20-50 trades/year (fee drag manageable at 0.05% RT)
- Filters 15m/1h noise while catching major moves
- Proven in baseline (mtf_hma_rsi_zscore_v1 Sharpe=5.4)

Key components:
1. Choppiness Index(14) for regime detection
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. HMA(21) on 4h for trend, HMA(21) on 1d for HTF bias
4. ATR(14) trailing stoploss at 2.5x
5. Discrete position sizing: 0.0, ±0.25, ±0.30

Timeframe: 4h (REQUIRED for experiment #001)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_chop_connors_1d_filter_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index for regime detection.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        choppiness = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    choppiness = np.nan_to_num(choppiness, nan=50.0, posinf=50.0, neginf=50.0)
    choppiness = np.clip(choppiness, 0, 100)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI for mean reversion entries.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    CRSI < 10 = oversold (long)
    CRSI > 90 = overbought (short)
    """
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / avg_streak_loss
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    # Percent Rank component
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50,
        raw=False
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # Combine components
    connors_rsi = (rsi_short + rsi_streak + percent_rank) / 3
    connors_rsi = np.clip(connors_rsi, 0, 100)
    
    return connors_rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, 21)
    choppiness = calculate_choppiness(high, low, close, 14)
    connors_rsi = calculate_connors_rsi(close, 3, 2, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    TREND_SIZE = 0.30
    MEAN_REVERT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(choppiness[i]):
            continue
        
        if np.isnan(connors_rsi[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_21[i] > hma_4h_21[i-1] if i > 0 else False
        hma_bearish = hma_4h_21[i] < hma_4h_21[i-1] if i > 0 else False
        
        # === REGIME DETECTION ===
        is_choppy = choppiness[i] > 61.8
        is_trending = choppiness[i] < 38.2
        is_neutral = not is_choppy and not is_trending
        
        # === MEAN REVERSION SIGNALS (choppy regime) ===
        crsi_oversold = connors_rsi[i] < 15
        crsi_overbought = connors_rsi[i] > 85
        
        # === TREND FOLLOWING SIGNALS (trending regime) ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI FILTER ===
        rsi_ok_long = rsi_14[i] < 70
        rsi_ok_short = rsi_14[i] > 30
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        if is_choppy:
            # MEAN REVERSION MODE
            # Long: CRSI oversold + price above 1d HMA + RSI not extreme
            if crsi_oversold and daily_bullish and rsi_ok_long:
                new_signal = MEAN_REVERT_SIZE
            # Short: CRSI overbought + price below 1d HMA + RSI not extreme
            elif crsi_overbought and daily_bearish and rsi_ok_short:
                new_signal = -MEAN_REVERT_SIZE
        
        elif is_trending:
            # TREND FOLLOWING MODE
            # Long: HMA bullish + breakout + daily bias confirms
            if hma_bullish and breakout_long and daily_bullish:
                new_signal = TREND_SIZE
            # Short: HMA bearish + breakout + daily bias confirms
            elif hma_bearish and breakout_short and daily_bearish:
                new_signal = -TREND_SIZE
        
        else:
            # NEUTRAL REGIME - use weaker signals
            # Allow entries with just CRSI extremes (no regime confirmation needed)
            if crsi_oversold and daily_bullish:
                new_signal = MEAN_REVERT_SIZE * 0.7
            elif crsi_overbought and daily_bearish:
                new_signal = -MEAN_REVERT_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~13 days on 4h), allow weaker entries
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if daily_bullish and connors_rsi[i] < 30:
                new_signal = BASE_SIZE * 0.6
            elif daily_bearish and connors_rsi[i] > 70:
                new_signal = -BASE_SIZE * 0.6
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        regime_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and is_trending and not hma_bullish:
                regime_exit = True
            if position_side < 0 and is_trending and not hma_bearish:
                regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            else:
                # Same direction, maintain position
                pass
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals