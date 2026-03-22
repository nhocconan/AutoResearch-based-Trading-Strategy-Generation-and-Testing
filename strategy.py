#!/usr/bin/env python3
"""
Experiment #100: 1h Primary + 4h/12h HTF — Regime-Adaptive Connors RSI with Session Filter

Hypothesis: Previous 1h strategies failed because they either generated too many trades
(>200/year → fee drag) or too few (0 trades → Sharpe=0). This strategy uses:
1. 12h Choppiness Index for regime detection (range vs trend)
2. 4h HMA(21) for major trend bias (only trade with HTF trend)
3. 1h Connors RSI for precise entry timing (pullback entries)
4. Session filter (8-20 UTC) to avoid low-liquidity periods
5. Volume filter (>0.8x 20-bar average) to confirm participation
6. Discrete position sizing (0.25 base) with ATR trailing stoploss

Why this should work:
- 12h CHOP filters out whipsaw regimes (major cause of BTC/ETH losses)
- 4h HMA prevents counter-trend trades in strong moves
- 1h CRSI catches pullbacks within HTF trend (75% win rate in literature)
- Session filter avoids Asian session low-volume traps
- Volume confirmation prevents false breakouts
- Target: 40-80 trades/year (within 1h fee drag limits)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for 1h TF)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_connors_session_4h12h_v1"
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 12h indicators
    chop_12h = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        14
    )
    hma_12h_50 = calculate_hma(df_12h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    hma_1h_21 = calculate_hma(close, 21)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    BASE_SIZE = 0.25
    
    # Track position state
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(chop_12h_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_1h_21[i]) or np.isnan(vol_avg[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
        volume_confirmed = vol_ratio > 0.8
        
        # === 12H REGIME (CHOPPINESS) ===
        is_range_market = chop_12h_aligned[i] > 55
        is_trend_market = chop_12h_aligned[i] < 45
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.3
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.3
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 12H MAJOR TREND ===
        price_above_12h_hma = close[i] > hma_12h_50_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_50_aligned[i]
        
        # === 1H CRSI SIGNALS ===
        crsi_oversold = crsi[i] < 18
        crsi_overbought = crsi[i] > 82
        crsi_pullback_low = crsi[i] < 35
        crsi_pullback_high = crsi[i] > 65
        
        # === 1H TREND CONFIRMATION ===
        hma_1h_bullish = hma_1h_21[i] > hma_4h_21_aligned[i] if not np.isnan(hma_4h_21_aligned[i]) else False
        hma_1h_bearish = hma_1h_21[i] < hma_4h_21_aligned[i] if not np.isnan(hma_4h_21_aligned[i]) else False
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_confluence = 0
        
        # Confluence 1: HTF trend bias
        if trend_4h_bullish or price_above_4h_hma:
            long_confluence += 1
        
        # Confluence 2: 12H major trend
        if price_above_12h_hma:
            long_confluence += 1
        
        # Confluence 3: CRSI entry signal
        if is_range_market and crsi_oversold:
            long_confluence += 2  # Stronger signal in range
        elif is_trend_market and crsi_pullback_low:
            long_confluence += 1
        elif crsi[i] < 25:
            long_confluence += 1
        
        # Confluence 4: Session filter
        if in_session:
            long_confluence += 1
        
        # Confluence 5: Volume confirmation
        if volume_confirmed:
            long_confluence += 1
        
        # Need 3+ confluence for entry
        if long_confluence >= 3:
            new_signal = current_size
        
        # SHORT ENTRIES
        short_confluence = 0
        
        if trend_4h_bearish or price_below_4h_hma:
            short_confluence += 1
        
        if price_below_12h_hma:
            short_confluence += 1
        
        if is_range_market and crsi_overbought:
            short_confluence += 2
        elif is_trend_market and crsi_pullback_high:
            short_confluence += 1
        elif crsi[i] > 75:
            short_confluence += 1
        
        if in_session:
            short_confluence += 1
        
        if volume_confirmed:
            short_confluence += 1
        
        if short_confluence >= 3:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and crsi[i] < 35 and in_session:
                new_signal = current_size * 0.5
            elif trend_4h_bearish and crsi[i] > 65 and in_session:
                new_signal = -current_size * 0.5
        
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
            # Exit long if regime becomes strongly trending bearish
            if position_side > 0 and is_trend_market and trend_4h_bearish and price_below_4h_hma:
                regime_reversal = True
            # Exit short if regime becomes strongly trending bullish
            if position_side < 0 and is_trend_market and trend_4h_bullish and price_above_4h_hma:
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