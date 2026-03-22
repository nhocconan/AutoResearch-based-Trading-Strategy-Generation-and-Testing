#!/usr/bin/env python3
"""
Experiment #159: 4h Primary + 1d HTF — KAMA Adaptive Trend + Connors RSI + Choppiness Regime

Hypothesis: Previous vol-spike strategies failed because capitulation events are too rare.
Research shows KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency better
than EMA/HMA in crypto's mixed regime environment. Combined with Connors RSI for entry
timing and Choppiness Index for regime detection, this should work across bull/bear/range.

Key innovations vs failed #149:
1. KAMA instead of HMA — adapts speed based on market noise (ER ratio)
2. Lower Connors RSI thresholds (20/80 vs 15/85) for MORE trades
3. Multiple entry paths — any 2 of 4 conditions triggers (not all required)
4. 1d HMA slope only for bias, not hard filter (allows counter-trend in ranges)
5. Reduced min hold bars (40 vs 80) for more turnover

Why 4h + 1d:
- 4h = 20-50 trades/year target (proven sweet spot)
- 1d HTF prevents fighting major trends but allows range mean-reversion
- KAMA adapts to crypto's shifting volatility regimes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.2 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_connors_chop_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency ratio.
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(close_s.diff(er_period))
    sum_changes = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / sum_changes.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.2)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    kama_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # KAMA trend direction
    kama_trend = np.zeros(n)
    for i in range(1, n):
        if kama_20[i] > kama_20[i-1]:
            kama_trend[i] = 1
        elif kama_20[i] < kama_20[i-1]:
            kama_trend[i] = -1
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(kama_20[i]):
            continue
        
        # === 1D TREND BIAS (soft filter, not hard) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === KAMA TREND ===
        kama_bullish = kama_trend[i] > 0
        kama_bearish = kama_trend[i] < 0
        price_above_kama = close[i] > kama_20[i]
        price_below_kama = close[i] < kama_20[i]
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        crsi_neutral_low = crsi[i] < 35
        crsi_neutral_high = crsi[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_range_market:
            current_size = BASE_SIZE * 1.1  # More aggressive in ranges
        elif is_trend_market:
            current_size = BASE_SIZE * 0.9  # More conservative in trends
        
        # === ENTRY LOGIC — Multiple paths for MORE trades ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES — Any 2 of 4 conditions triggers
        long_conditions = 0
        
        # Condition 1: CRSI oversold
        if crsi_oversold or crsi_extreme_low:
            long_conditions += 1
        
        # Condition 2: Price below BB lower (oversold)
        if price_below_bb_lower or bb_pct < 0.15:
            long_conditions += 1
        
        # Condition 3: KAMA bullish or price above KAMA
        if kama_bullish or price_above_kama:
            long_conditions += 1
        
        # Condition 4: 1d bias not strongly bearish
        if trend_1d_bullish or (not trend_1d_bearish and is_range_market):
            long_conditions += 1
        
        # Need 2+ conditions for full size, 1+ for half size with cooldown
        if long_conditions >= 2:
            new_signal = current_size
        elif long_conditions >= 1 and bars_since_last_trade > 40:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Condition 1: CRSI overbought
        if crsi_overbought or crsi_extreme_high:
            short_conditions += 1
        
        # Condition 2: Price above BB upper (overbought)
        if price_above_bb_upper or bb_pct > 0.85:
            short_conditions += 1
        
        # Condition 3: KAMA bearish or price below KAMA
        if kama_bearish or price_below_kama:
            short_conditions += 1
        
        # Condition 4: 1d bias not strongly bullish
        if trend_1d_bearish or (not trend_1d_bullish and is_range_market):
            short_conditions += 1
        
        if short_conditions >= 2:
            new_signal = -current_size
        elif short_conditions >= 1 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD — Force trades if dormant ===
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if crsi_extreme_low and price_below_bb_lower:
                new_signal = current_size * 0.4
            elif crsi_extreme_high and price_above_bb_upper:
                new_signal = -current_size * 0.4
            elif crsi_neutral_low and trend_1d_bullish:
                new_signal = current_size * 0.3
            elif crsi_neutral_high and trend_1d_bearish:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.2 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.2 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.2 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 70:
                crsi_reversal = True
            if position_side < 0 and crsi[i] < 30:
                crsi_reversal = True
        
        if stoploss_triggered or crsi_reversal:
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