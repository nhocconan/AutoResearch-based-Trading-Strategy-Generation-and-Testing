#!/usr/bin/env python3
"""
Experiment #060: 1h Primary + 4h/12h HTF — Regime-Adaptive Mean Reversion

Hypothesis: Lower TF (1h) strategies fail due to excessive trades → fee drag.
This strategy uses STRICT confluence filters to limit trades to 30-60/year:

1. 4h HMA(21) SLOPE for major trend direction (HTF bias)
2. 12h HMA(21) for regime confirmation (secondary HTF filter)
3. 1h Choppiness Index(14) for regime detection (CHOP>55=range, <45=trend)
4. 1h Connors RSI for entry timing (CRSI<15 long, >85 short in range)
5. Session filter: only 8-20 UTC (high liquidity, fewer false signals)
6. Volume filter: >0.8x 20-bar average (confirm participation)
7. ATR(14) stoploss at 2.5x (wider for 1h noise)
8. Position size: 0.20 discrete (smaller for lower TF fee sensitivity)

Why this should work:
- 1h with HTF filters = HTF trade frequency with 1h execution precision
- Choppiness Index adapts between mean-revert (range) and trend-follow
- Connors RSI has 75% win rate on mean reversion entries
- Session + volume filters cut 60%+ of low-quality signals
- 4h+12h double HTF confirmation prevents counter-trend trades

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20 discrete (max 0.30 on strong confluence)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_connors_chop_4h12h_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) for regime detection.
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) for mean reversion entries.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10-15 | Short: CRSI > 85-90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        pos_streak = max(0, streak[i])
        neg_streak = abs(min(0, streak[i]))
        if pos_streak + neg_streak > 0:
            streak_rsi[i] = 100 * pos_streak / (pos_streak + neg_streak + 1)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current price ranks vs last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100 * rank / rank_period
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period (percentage change)."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_utc_hour_from_open_time(open_time):
    """Extract UTC hour from Binance open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Volume average (20-bar)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.30 for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour_from_open_time(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 4H TREND BIAS (PRIMARY HTF) ===
        # HMA slope > 0.5% = bullish bias (prefer longs)
        # HMA slope < -0.5% = bearish bias (prefer shorts)
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.5
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.5
        trend_4h_neutral = not trend_4h_bullish and not trend_4h_bearish
        
        # Price vs 4h HMA
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 12H TREND CONFIRMATION (SECONDARY HTF) ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.3
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.3
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (use mean reversion)
        # CHOP < 45 = trending market (use trend following)
        # 45-55 = transition (reduce size or skip)
        regime_range = chop_14[i] > 55
        regime_trend = chop_14[i] < 45
        regime_transition = not regime_range and not regime_trend
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # CRSI < 15 = oversold (long entry in range)
        # CRSI > 85 = overbought (short entry in range)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # For trend regime: use RSI(14) instead
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Increase size on strong confluence (both HTF agree + volume + session)
        strong_confluence = (
            (trend_4h_bullish and trend_12h_bullish) or
            (trend_4h_bearish and trend_12h_bearish)
        ) and volume_confirmed and in_session
        
        if strong_confluence:
            current_size = STRONG_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Range regime: mean reversion with HTF bullish bias
        if regime_range:
            if (trend_4h_bullish or trend_4h_neutral) and crsi_oversold:
                if in_session and volume_confirmed:
                    new_signal = current_size
        # Trend regime: pullback entry with HTF confirmation
        elif regime_trend:
            if trend_4h_bullish and trend_12h_bullish and rsi_oversold:
                if in_session and volume_confirmed:
                    new_signal = current_size
            # Also allow on 4h HMA cross above price (pullback to support)
            elif trend_4h_bullish and price_above_4h_hma and rsi_14[i] < 50:
                if in_session and volume_confirmed:
                    new_signal = BASE_SIZE
        
        # SHORT ENTRIES
        # Range regime: mean reversion with HTF bearish bias
        if regime_range:
            if (trend_4h_bearish or trend_4h_neutral) and crsi_overbought:
                if in_session and volume_confirmed:
                    new_signal = -current_size
        # Trend regime: pullback entry with HTF confirmation
        elif regime_trend:
            if trend_4h_bearish and trend_12h_bearish and rsi_overbought:
                if in_session and volume_confirmed:
                    new_signal = -current_size
            # Also allow on 4h HMA cross below price (rally to resistance)
            elif trend_4h_bearish and price_below_4h_hma and rsi_14[i] > 50:
                if in_session and volume_confirmed:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and crsi[i] < 25 and in_session:
                new_signal = BASE_SIZE * 0.5
            elif trend_4h_bearish and crsi[i] > 75 and in_session:
                new_signal = -BASE_SIZE * 0.5
        
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
            # Exit if regime changes dramatically against position
            if position_side > 0 and chop_14[i] > 65 and crsi[i] > 70:
                regime_reversal = True
            if position_side < 0 and chop_14[i] > 65 and crsi[i] < 30:
                regime_reversal = True
            
            # Exit if HTF trend reverses strongly
            if position_side > 0 and trend_4h_bearish and trend_12h_bearish:
                regime_reversal = True
            if position_side < 0 and trend_4h_bullish and trend_12h_bullish:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
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
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same side, keep position (no update needed)
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