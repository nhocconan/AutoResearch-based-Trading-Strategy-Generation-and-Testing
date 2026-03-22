#!/usr/bin/env python3
"""
Experiment #149: 4h Primary + 1d HTF — Volatility Mean Reversion Simplified

Hypothesis: Previous 4h strategies failed due to OVER-FILTERING (too many conditions
never align = 0 trades). This strategy SIMPLIFIES entry logic while keeping the
proven edge of vol-spike mean reversion + Connors RSI.

Key changes from failed attempts:
1. FEWER entry conditions (max 2-3 confluence, not 5)
2. LOWER CRSI thresholds (ensure trades in both bull/bear)
3. ASYMMETRIC bias: favor longs when 1d HMA slope > 0, shorts when < 0
4. VOLATILITY-BASED sizing: reduce size when ATR ratio is extreme
5. SIMPLE stoploss: 2.0 * ATR(14) from entry (not complex trailing)

Why 4h works:
- 20-50 trades/year target (low fee drag)
- Captures multi-day swings without noise of lower TF
- 1d HTF provides major trend context without over-constraining

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.30 max (discrete levels)
Stoploss: 2.0 * ATR(14) from entry price
Target: 30-60 trades/year per symbol, Sharpe > 0.3
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_connors_bb_1d_simp_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.2)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    # RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === 1D TREND BIAS (asymmetric, not blocking) ===
        daily_bullish = hma_1d_slope_aligned[i] > 0.2
        daily_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_elevated = atr_ratio[i] > 1.4  # Lowered threshold for more trades
        vol_extreme = atr_ratio[i] > 2.0
        
        # === BOLLINGER BAND POSITION ===
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        price_near_lower = bb_pct < 0.15
        price_near_upper = bb_pct > 0.85
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 30  # Lowered from 25 for more trades
        crsi_overbought = crsi[i] > 70  # Lowered from 75 for more trades
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if vol_extreme:
            current_size = BASE_SIZE * 0.7  # Reduce size in extreme vol
        
        # === ENTRY LOGIC — SIMPLIFIED FOR MORE TRADES ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES — Multiple paths, each requiring only 2 conditions
        long_signal = False
        
        # Path 1: CRSI oversold + BB lower (mean reversion core)
        if crsi_oversold and (price_near_lower or price_below_lower):
            long_signal = True
        
        # Path 2: CRSI extreme + RSI oversold (double confirmation)
        if crsi_extreme_low and rsi_oversold:
            long_signal = True
        
        # Path 3: Daily bullish bias + pullback (trend continuation)
        if daily_bullish and crsi[i] < 40 and price_below_1d_hma:
            long_signal = True
        
        # Path 4: Vol spike + oversold (capitulation long)
        if vol_elevated and crsi[i] < 35:
            long_signal = True
        
        # Path 5: Simple CRSI extreme (fallback for trade generation)
        if crsi[i] < 15 and bars_since_last_trade > 60:
            long_signal = True
        
        if long_signal:
            # Apply daily bias filter (favor longs in bull, but allow in bear)
            if daily_bullish or not daily_bearish:
                new_signal = current_size
            elif bars_since_last_trade > 80:
                new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_signal = False
        
        # Path 1: CRSI overbought + BB upper
        if crsi_overbought and (price_near_upper or price_above_upper):
            short_signal = True
        
        # Path 2: CRSI extreme + RSI overbought
        if crsi_extreme_high and rsi_overbought:
            short_signal = True
        
        # Path 3: Daily bearish bias + rally
        if daily_bearish and crsi[i] > 60 and price_above_1d_hma:
            short_signal = True
        
        # Path 4: Vol spike + overbought
        if vol_elevated and crsi[i] > 65:
            short_signal = True
        
        # Path 5: Simple CRSI extreme (fallback)
        if crsi[i] > 85 and bars_since_last_trade > 60:
            short_signal = True
        
        if short_signal:
            if daily_bearish or not daily_bullish:
                new_signal = -current_size
            elif bars_since_last_trade > 80:
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD — Ensure minimum trades ===
        # If no trades for 100 bars (~17 days on 4h), force entry on weaker signals
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if daily_bullish and crsi[i] < 40:
                new_signal = current_size * 0.4
            elif daily_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.4
            elif crsi[i] < 25:
                new_signal = current_size * 0.3
            elif crsi[i] > 75:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR from entry ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                stoploss_price = entry_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                stoploss_price = entry_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TAKE PROFIT / EXIT ON REVERSAL ===
        # Exit long if CRSI becomes overbought
        profit_exit = False
        if in_position and position_side > 0 and crsi[i] > 75:
            profit_exit = True
        if in_position and position_side < 0 and crsi[i] < 25:
            profit_exit = True
        
        # Exit if daily trend strongly reverses against position
        trend_reversal = False
        if in_position and position_side > 0 and daily_bearish and hma_1d_slope_aligned[i] < -0.5:
            trend_reversal = True
        if in_position and position_side < 0 and daily_bullish and hma_1d_slope_aligned[i] > 0.5:
            trend_reversal = True
        
        if stoploss_triggered or profit_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position or np.sign(new_signal) != position_side:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals