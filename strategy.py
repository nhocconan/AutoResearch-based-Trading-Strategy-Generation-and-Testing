#!/usr/bin/env python3
"""
Experiment #015: 1h Regime-Adaptive CRSI + CHOP + Multi-HTF Trend

Hypothesis: Previous 1h/30m strategies failed because they used single-regime logic
(either always trend-follow or always mean-revert). Crypto alternates between
trending and ranging phases. This strategy adapts:

1. CHOPPINESS INDEX (14) - Regime detector
   CHOP > 55 = Range regime → use mean-reversion (CRSI extremes)
   CHOP < 45 = Trend regime → use trend-follow (pullback to HTF HMA)
   CHOP 45-55 = Neutral → reduce position size or stay flat

2. CONNORS RSI (CRSI) - Entry timing for range regime
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long: CRSI < 15 (oversold in range)
   Short: CRSI > 85 (overbought in range)
   Higher win rate than standard RSI for mean-reversion

3. 4h HMA(21) - Intermediate trend via mtf_data helper
   Only long if price > 4h HMA (in range regime)
   Only short if price < 4h HMA (in range regime)

4. 1d HMA(21) - Major trend bias via mtf_data helper
   Increases position size when 4h and 1d align (high conviction)
   Reduces size when they diverge

5. SESSION FILTER - Only trade UTC 8-20 (high liquidity hours)
   Avoids Asian session whipsaws and low-volume periods

6. VOLUME FILTER - Volume > 0.8x 20-bar average
   Confirms genuine moves, not low-liquidity noise

7. ATR(14) Trailing Stop - 2.5x ATR for risk management

Timeframe: 1h (REQUIRED for Experiment #015)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Target trades: 30-60/year (use strict confluence to avoid fee drag)
Position sizing: 0.20 base, 0.30 high conviction, 0.15 low conviction
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_chop_4h_1d_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / hh_ll.replace(0, np.nan)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines RSI, streak RSI, and percent rank.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    Streak: consecutive up/down days
    PercentRank: where current close ranks in last N closes
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_close = 100 - (100 / (1 + rs))
    rsi_close = rsi_close.replace([np.inf, -np.inf], np.nan)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.replace([np.inf, -np.inf], np.nan)
    
    # Percent Rank - where current close ranks in last rank_period closes
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50,
        raw=False
    )
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    crsi = crsi.replace([np.inf, -np.inf], np.nan)
    
    return crsi.values

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    volume_s = pd.Series(volume)
    vol_avg = volume_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume_s / vol_avg.replace(0, np.nan)
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
    return vol_ratio.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts = pd.to_datetime(open_time, unit='ms', utc=True)
    return ts.dt.hour.values

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
    
    # Calculate 4h HMA for intermediate trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.20
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.15
    
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
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === SESSION FILTER - Only trade UTC 8-20 ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === REGIME DETECTION ===
        chop_value = chop_14[i]
        range_regime = chop_value > 55  # Mean-reversion mode
        trend_regime = chop_value < 45  # Trend-follow mode
        neutral_regime = 45 <= chop_value <= 55
        
        # === TREND FILTERS ===
        above_4h_hma = close[i] > hma_4h_21_aligned[i]
        below_4h_hma = close[i] < hma_4h_21_aligned[i]
        above_1d_hma = close[i] > hma_1d_21_aligned[i]
        below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_moderate_oversold = crsi[i] < 25
        crsi_moderate_overbought = crsi[i] > 75
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # RANGE REGIME - Mean Reversion (CRSI extremes + HTF trend filter)
        if range_regime and in_session and volume_ok:
            # LONG: CRSI oversold + price above 4h HMA (trend-aligned mean reversion)
            if crsi_oversold and above_4h_hma:
                if above_1d_hma:
                    new_signal = HIGH_CONV_SIZE  # 0.30 - high conviction
                else:
                    new_signal = BASE_SIZE  # 0.20 - base
            elif crsi_moderate_oversold and above_4h_hma and above_1d_hma:
                new_signal = BASE_SIZE
            
            # SHORT: CRSI overbought + price below 4h HMA
            if crsi_overbought and below_4h_hma:
                if below_1d_hma:
                    new_signal = -HIGH_CONV_SIZE  # -0.30 - high conviction
                else:
                    new_signal = -BASE_SIZE  # -0.20 - base
            elif crsi_moderate_overbought and below_4h_hma and below_1d_hma:
                new_signal = -BASE_SIZE
        
        # TREND REGIME - Trend Following (pullback to 4h HMA)
        elif trend_regime and in_session and volume_ok:
            # LONG: Price above both HTF HMAs + pullback near 4h HMA
            if above_4h_hma and above_1d_hma:
                # Pullback: price within 1% of 4h HMA
                pullback_long = abs(close[i] - hma_4h_21_aligned[i]) / hma_4h_21_aligned[i] < 0.01
                if pullback_long and crsi_moderate_oversold:
                    new_signal = HIGH_CONV_SIZE
            
            # SHORT: Price below both HTF HMAs + bounce near 4h HMA
            if below_4h_hma and below_1d_hma:
                pullback_short = abs(close[i] - hma_4h_21_aligned[i]) / hma_4h_21_aligned[i] < 0.01
                if pullback_short and crsi_moderate_overbought:
                    new_signal = -HIGH_CONV_SIZE
        
        # NEUTRAL REGIME - Reduced sizing or flat
        elif neutral_regime and in_session and volume_ok:
            # Only take high-conviction setups
            if crsi_oversold and above_4h_hma and above_1d_hma:
                new_signal = LOW_CONV_SIZE
            elif crsi_overbought and below_4h_hma and below_1d_hma:
                new_signal = -LOW_CONV_SIZE
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position and in_session:
            if above_4h_hma and above_1d_hma and crsi_moderate_oversold:
                new_signal = LOW_CONV_SIZE
            elif below_4h_hma and below_1d_hma and crsi_moderate_overbought:
                new_signal = -LOW_CONV_SIZE
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit long if regime shifts to strong trend bearish
            if position_side > 0 and trend_regime and below_4h_hma:
                regime_exit = True
            # Exit short if regime shifts to strong trend bullish
            if position_side < 0 and trend_regime and above_4h_hma:
                regime_exit = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_exit or crsi_exit:
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