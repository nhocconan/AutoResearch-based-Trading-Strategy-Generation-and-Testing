#!/usr/bin/env python3
"""
Experiment #115: 1h Primary + 4h/1d HTF — Regime-Adaptive Connors RSI with Session Filter

Hypothesis: Previous 1h strategies failed because they either generated 0 trades (too strict)
or too many trades (>200/year → fee drag). This strategy uses:

1. 4h HMA(21) SLOPE: Primary trend direction (only trade WITH HTF trend)
2. 1d HMA(21): Major trend confirmation (avoid counter-trend in strong moves)
3. CONNORS RSI(3,2,100): Entry timing at extremes (<25 long, >75 short)
4. CHOPPINESS INDEX(14): Regime filter (range>55 = mean revert, trend<45 = pullback)
5. SESSION FILTER: Only trade 8-20 UTC (high liquidity hours, avoid Asian chop)
6. VOLUME FILTER: Volume > 0.8x 20-bar average (confirm participation)
7. ATR STOPLOSS: 2.5 * ATR(14) trailing stop

Why this should work:
- 4h/1d HTF ensures we trade with major trend (reduces whipsaw)
- Connors RSI has 75% win rate for mean reversion in literature
- Session filter reduces trades by ~60% (only 12/24 hours)
- 1h timeframe with strict filters = 30-60 trades/year target
- Discrete position sizing (0.20) minimizes fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20 discrete (smaller for lower TF per rules)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_regime_session_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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
    
    # Convert streak to 0-100 scale
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 12.5)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 12.5)
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50,
        raw=False
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

def extract_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    open_time_ms = prices["open_time"].values
    # Convert to hours UTC: (ms / 1000 / 3600) % 24
    hours_utc = ((open_time_ms / 1000 / 3600) % 24).astype(int)
    return hours_utc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hour for session filter
    hours_utc = extract_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, lower for 1h TF)
    BASE_SIZE = 0.20
    
    # Track position state
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours_utc[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === 4H TREND BIAS (primary direction filter) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D TREND CONFIRMATION ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.1
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.1
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 15
        crsi_extreme_high = crsi[i] > 85
        crsi_moderate_low = crsi[i] < 35
        crsi_moderate_high = crsi[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Require session + volume for all entries (reduces trades significantly)
        entry_allowed = in_session and volume_ok
        
        if entry_allowed:
            # LONG ENTRIES - Multiple confluence paths
            long_confidence = 0
            
            # Path 1: Range market + CRSI extreme oversold (mean revert)
            if is_range_market and crsi_extreme_low:
                long_confidence += 3
            
            # Path 2: 4h bullish trend + pullback + CRSI moderate low
            if trend_4h_bullish and price_above_4h_hma and crsi_moderate_low:
                long_confidence += 2
            
            # Path 3: 4h + 1d both bullish + CRSI oversold (strong confluence)
            if trend_4h_bullish and trend_1d_bullish and crsi_oversold:
                long_confidence += 3
            
            # Path 4: Price above 4h HMA + CRSI very low (deep pullback in uptrend)
            if price_above_4h_hma and crsi[i] < 20:
                long_confidence += 2
            
            # Path 5: 4h bullish + CRSI moderate (simpler entry for more trades)
            if trend_4h_bullish and crsi[i] < 30:
                long_confidence += 1
            
            if long_confidence >= 3:
                new_signal = current_size
            elif long_confidence == 2 and bars_since_last_trade > 60:
                new_signal = current_size * 0.75
            elif long_confidence == 1 and bars_since_last_trade > 100:
                new_signal = current_size * 0.5
            
            # SHORT ENTRIES
            short_confidence = 0
            
            # Path 1: Range market + CRSI extreme overbought
            if is_range_market and crsi_extreme_high:
                short_confidence += 3
            
            # Path 2: 4h bearish trend + pullback + CRSI moderate high
            if trend_4h_bearish and price_below_4h_hma and crsi_moderate_high:
                short_confidence += 2
            
            # Path 3: 4h + 1d both bearish + CRSI overbought
            if trend_4h_bearish and trend_1d_bearish and crsi_overbought:
                short_confidence += 3
            
            # Path 4: Price below 4h HMA + CRSI very high (rally in downtrend)
            if price_below_4h_hma and crsi[i] > 80:
                short_confidence += 2
            
            # Path 5: 4h bearish + CRSI moderate (simpler entry)
            if trend_4h_bearish and crsi[i] > 70:
                short_confidence += 1
            
            if short_confidence >= 3:
                new_signal = -current_size
            elif short_confidence == 2 and bars_since_last_trade > 60:
                new_signal = -current_size * 0.75
            elif short_confidence == 1 and bars_since_last_trade > 100:
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # If no trade for 200 bars (~8 days on 1h), relax conditions
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and crsi[i] < 35 and in_session:
                new_signal = current_size * 0.5
            elif trend_4h_bearish and crsi[i] > 65 and in_session:
                new_signal = -current_size * 0.5
            elif crsi_extreme_low and in_session:
                new_signal = current_size * 0.4
            elif crsi_extreme_high and in_session:
                new_signal = -current_size * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and trend_4h_bearish:
                regime_reversal = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and trend_4h_bullish:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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