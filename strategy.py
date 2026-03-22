#!/usr/bin/env python3
"""
Experiment #005: 1h Multi-Timeframe Regime-Adaptive Strategy

Hypothesis: 1h timeframe with strict confluence filters can capture HTF trend moves
while minimizing fee drag through session/volume filters. Key innovations:
1. 4h HMA for major trend direction (HTF bias)
2. 1d HMA for macro regime filter (bull/bear market)
3. Choppiness Index (14) to switch between trend/mean-reversion logic
4. Connors RSI for precise mean-reversion entries in range regime
5. Session filter (8-20 UTC) to avoid low-liquidity Asian hours
6. Volume filter (>0.8x 20-bar avg) to confirm breakout validity
7. ATR stoploss (2.5x) with discrete position sizing (0.25-0.30)

Why this should work on 1h:
- HTF (4h/1d) provides signal DIRECTION, 1h only for ENTRY TIMING
- Session + volume filters reduce trades to 30-80/year target
- Regime-adaptive logic works in both bull (2021) and bear (2025) markets
- Discrete signal levels minimize fee churn from tiny position changes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
Trade frequency target: 30-80 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_chop_connors_session_vol_4h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Range/Consolidation
    CHOP < 38.2 = Trending
    """
    atr = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    rsi_close = calculate_rsi(close, rsi_period)
    
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_positive = np.where(streak > 0, streak, 0)
    streak_negative = np.where(streak < 0, -streak, 0)
    
    streak_avg_gain = pd.Series(streak_positive).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_avg_loss = pd.Series(streak_negative).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_avg_loss = np.where(streak_avg_loss == 0, 1e-10, streak_avg_loss)
    streak_rs = streak_avg_gain / streak_avg_loss
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.nan_to_num(rsi_streak, nan=50.0)
    
    returns = close_s.pct_change().values
    percent_rank = np.full(n, 50.0)
    
    for i in range(rank_period, n):
        if not np.isnan(returns[i]):
            window = returns[max(0, i-rank_period):i]
            window = window[~np.isnan(window)]
            if len(window) > 0:
                percent_rank[i] = 100 * np.sum(window < returns[i]) / len(window)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    if isinstance(open_time[0], (int, np.integer)):
        hours = (open_time // 3600000) % 24
    else:
        hours = pd.to_datetime(open_time).hour.values
    return hours

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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1h_16 = calculate_hma(close, 16)
    hma_1h_48 = calculate_hma(close, 48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume moving average for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (8-20 UTC)
    hours = extract_hour(open_time)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_1h_16[i]) or np.isnan(hma_1h_48[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === SESSION FILTER (Rule 10 - reduce trades on lower TF) ===
        if not in_session[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
            continue
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_ma_20[i]
        volume_confirmed = vol_ratio > 0.75
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop_14[i] > 58.0
        is_trend = chop_14[i] < 42.0
        
        # === HTF TREND BIAS (4h and 1d) ===
        hma_4h_bullish = close[i] > hma_4h_21_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        hma_1d_bullish = close[i] > hma_1d_21_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 1H HMA TREND ===
        hma_1h_bullish = hma_1h_16[i] > hma_1h_48[i]
        hma_1h_bearish = hma_1h_16[i] < hma_1h_48[i]
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE + 3+ CONFLUENCE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # RANGE REGIME: Connors RSI Mean Reversion (4+ confluence)
        if is_range:
            # LONG: CRSI<12 + 4h not bearish + 1d not bearish + volume confirmed
            if crsi[i] < 12 and not hma_4h_bearish and not hma_1d_bearish and volume_confirmed:
                new_signal = current_size
            
            # SHORT: CRSI>88 + 4h not bullish + 1d not bullish + volume confirmed
            elif crsi[i] > 88 and not hma_4h_bullish and not hma_1d_bullish and volume_confirmed:
                new_signal = -current_size
        
        # TREND REGIME: HMA + Donchian Breakout (4+ confluence)
        elif is_trend:
            # LONG: 4h bullish + 1d bullish + 1h HMA bullish + Donchian breakout + volume
            if hma_4h_bullish and hma_1d_bullish and hma_1h_bullish and volume_confirmed:
                if i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1]:
                        new_signal = current_size
            
            # SHORT: 4h bearish + 1d bearish + 1h HMA bearish + Donchian breakout + volume
            if hma_4h_bearish and hma_1d_bearish and hma_1h_bearish and volume_confirmed:
                if i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1]:
                        new_signal = -current_size
        
        # NEUTRAL REGIME (42-58): Use simpler HMA alignment with HTF
        else:
            if hma_4h_bullish and hma_1d_bullish and hma_1h_bullish and volume_confirmed:
                new_signal = current_size * 0.7
            elif hma_4h_bearish and hma_1d_bearish and hma_1h_bearish and volume_confirmed:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 120 bars (~5 days on 1h), allow weaker entry
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and hma_1d_bullish and crsi[i] < 45:
                new_signal = current_size * 0.5
            elif hma_4h_bearish and hma_1d_bearish and crsi[i] > 55:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_4h_bearish and hma_1d_bearish:
                trend_reversal = True
            if position_side < 0 and hma_4h_bullish and hma_1d_bullish:
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