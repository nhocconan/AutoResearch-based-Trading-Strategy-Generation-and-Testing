#!/usr/bin/env python3
"""
Experiment #034: 4h Regime-Adaptive Choppiness + Connors RSI + HMA Trend

Hypothesis: Previous 4h strategies failed because they used pure trend-following
which gets whipsawed in range markets (most of 2022-2024). This strategy adapts
to market regime using Choppiness Index:

- CHOP > 61.8 (choppy/range): Connors RSI mean reversion at BB bounds
- CHOP < 38.2 (trending): HMA + Donchian breakout trend following
- 1d HMA(21) for major trend bias (only trade with daily trend)

Why this might work:
1. Regime detection prevents trend strategies in chop (where they fail)
2. Connors RSI has 75% win rate in mean reversion (research-backed)
3. 4h TF = 20-50 trades/year (fee drag manageable)
4. Discrete sizing (0.25/0.30) minimizes churn costs

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_connors_hma_donchian_1d_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean reversion indicator with 75% win rate.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        pos_count = np.sum(streak_window > 0)
        streak_rsi[i] = (pos_count / streak_period) * 100 if streak_period > 0 else 50
    
    # Percent Rank - where current price ranks in last N periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        current = close[i]
        rank = np.sum(window < current) / len(window) * 100 if len(window) > 0 else 50
        percent_rank[i] = rank
    
    # Fill early values
    rsi_short = np.nan_to_num(rsi_short, nan=50.0)
    streak_rsi = np.nan_to_num(streak_rsi, nan=50.0)
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    atr_values = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr_values[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

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
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness_index(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    bars_since_trade = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === BOLLINGER BAND POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002 if not np.isnan(bb_lower[i]) else False
        at_bb_upper = close[i] >= bb_upper[i] * 0.998 if not np.isnan(bb_upper[i]) else False
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === RSI FILTER ===
        rsi_ok_long = rsi_14[i] < 70
        rsi_ok_short = rsi_14[i] > 30
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.8, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.round(current_size * 4) / 4  # Round to 0.25 increments
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        if is_trending:
            # TREND FOLLOWING MODE
            # Long: HMA bullish + Donchian breakout + daily bias + RSI ok
            if hma_bullish and breakout_long and rsi_ok_long:
                if daily_bullish:
                    new_signal = current_size
                else:
                    new_signal = current_size * 0.7  # Smaller without daily confirmation
            
            # Short: HMA bearish + Donchian breakout + daily bias + RSI ok
            if hma_bearish and breakout_short and rsi_ok_short:
                if daily_bearish:
                    new_signal = -current_size
                else:
                    new_signal = -current_size * 0.7
        
        elif is_choppy:
            # MEAN REVERSION MODE
            # Long: Connors RSI oversold + at BB lower + daily bullish bias preferred
            if crsi_oversold and at_bb_lower:
                if daily_bullish:
                    new_signal = current_size
                elif not daily_bearish:  # Neutral daily
                    new_signal = current_size * 0.7
            
            # Short: Connors RSI overbought + at BB upper + daily bearish bias preferred
            if crsi_overbought and at_bb_upper:
                if daily_bearish:
                    new_signal = -current_size
                elif not daily_bullish:  # Neutral daily
                    new_signal = -current_size * 0.7
        
        else:
            # TRANSITIONAL REGIME - use simpler HMA crossover
            if hma_bullish and daily_bullish and rsi_ok_long:
                new_signal = current_size * 0.6
            if hma_bearish and daily_bearish and rsi_ok_short:
                new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~8 days on 4h), allow weaker entries
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            # Looser entry: just HMA + daily alignment
            if hma_bullish and daily_bullish and rsi_14[i] < 65:
                new_signal = current_size * 0.5
            elif hma_bearish and daily_bearish and rsi_14[i] > 35:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals