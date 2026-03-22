#!/usr/bin/env python3
"""
Experiment #295: 15m Mean Reversion with 4h Trend Bias and Connors RSI

Hypothesis: After 294 experiments, clear pattern emerges - 15m trend-following FAILS
(#283: -85.7%, #289: -91.6%) due to fee drag and whipsaws. However, 15m can work
for MEAN REVERSION entries within higher-timeframe trends.

This strategy combines:
1. 4h HMA(21) for directional bias (proven edge from #292)
2. 15m Connors RSI (CRSI) for oversold/overbought entries - 75% win rate in literature
3. Bollinger Band %B to confirm price at extremes
4. ADX(14) filter to avoid choppy markets (ADX>20)
5. ATR(14) trailing stoploss at 2.5*ATR (tighter for 15m)

Key innovation: Use 15m for counter-trend PULLBACK entries, not breakouts.
Long when: 4h trend up + 15m RSI(2)<10 + price at BB lower band
Short when: 4h trend down + 15m RSI(2)>90 + price at BB upper band

This is DIFFERENT from failed 15m strategies which tried trend-following.
Mean reversion on 15m within 4h trend should reduce whipsaws.

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing (tighter for 15m timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_meanrev_4h_hma_bb_adx_atr_v1"
timeframe = "15m"
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
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and %B."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    # %B = (price - lower) / (upper - lower)
    bb_pct = (close - lower) / (upper - lower)
    bb_pct = np.clip(bb_pct, 0.0, 1.0)
    return upper, lower, bb_pct

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals.
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    CRSI < 10 = oversold (long signal)
    CRSI > 90 = overbought (short signal)
    
    This has 75% win rate in literature for mean reversion.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi_short = 100 - (100 / (1 + rs))
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / streak_avg_loss
    streak_rs = streak_rs.replace([np.inf, -np.inf], np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Component 3: Percent Rank of recent returns
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        if len(window) > 0:
            percent_rank.iloc[i] = (window < returns.iloc[i]).sum() / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_2 = calculate_rsi(close, 2)  # Very short RSI for Connors
    rsi_14 = calculate_rsi(close, 14)  # Standard RSI for filter
    bb_upper, bb_lower, bb_pct = calculate_bollinger_bands(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_INCREASED = 0.30  # Increased size in strong trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_2[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_pct[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 20 = trending enough (not too choppy)
        trending = adx[i] > 20
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 10 = extremely oversold (long setup)
        # CRSI > 90 = extremely overbought (short setup)
        crsi_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 90
        
        # === RSI(2) EXTREMES ===
        # RSI(2) < 5 = very oversold
        # RSI(2) > 95 = very overbought
        rsi2_oversold = rsi_2[i] < 5
        rsi2_overbought = rsi_2[i] > 95
        
        # === BOLLINGER BAND POSITION ===
        # %B < 0.1 = price at lower band (oversold)
        # %B > 0.9 = price at upper band (overbought)
        bb_oversold = bb_pct[i] < 0.1
        bb_overbought = bb_pct[i] > 0.9
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Strong trend = increase size
        strong_trend = adx[i] > 30
        
        # Determine position size
        if high_volatility:
            position_size = SIZE_REDUCED
        elif strong_trend:
            position_size = SIZE_INCREASED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS (Mean Reversion within Trend) ===
        new_signal = 0.0
        
        # LONG: 4h trend up + 15m oversold (CRSI + RSI2 + BB confirm)
        # Need at least 2 of 3 oversold signals to trigger
        oversold_count = sum([crsi_oversold, rsi2_oversold, bb_oversold])
        
        long_conditions = (
            bull_trend_4h and  # 4h trend bullish
            oversold_count >= 2 and  # At least 2 oversold signals
            trending  # Not too choppy
        )
        
        # SHORT: 4h trend down + 15m overbought
        overbought_count = sum([crsi_overbought, rsi2_overbought, bb_overbought])
        
        short_conditions = (
            bear_trend_4h and  # 4h trend bearish
            overbought_count >= 2 and  # At least 2 overbought signals
            trending  # Not too choppy
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === MEAN REVERSION EXIT ===
        # Exit long when RSI(14) > 70 (overbought)
        # Exit short when RSI(14) < 30 (oversold)
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi_14[i] > 70:
                new_signal = 0.0  # Long target hit
            if position_side < 0 and rsi_14[i] < 30:
                new_signal = 0.0  # Short target hit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals