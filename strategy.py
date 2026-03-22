#!/usr/bin/env python3
"""
Experiment #408: 1d Weekly HMA Trend + Daily ADX/RSI Ensemble + BB Regime Filter

Hypothesis: After 407 failed experiments, the key insight is that DAILY timeframe
needs WEEKLY trend bias + ENSEMBLE voting + ASYMMETRIC entries.

WHY THIS SHOULD WORK (vs 407 failures):
1. WEEKLY HMA(21) is MORE STABLE than daily HMA - fewer false trend flips
2. ENSEMBLE VOTING (2 of 3 signals) reduces whipsaw vs single indicator
3. ASYMMETRIC entries: easier to enter WITH weekly trend, harder AGAINST
4. BB WIDTH percentile detects regime better than absolute CHOP values
5. LOW trade frequency target (25-40 trades/year) minimizes fee drag on 1d

STRATEGY COMPONENTS:
1. WEEKLY HMA(21) via mtf_data: Primary trend bias (bull/bear)
2. DAILY ADX(14): Trend strength filter (ADX>25 = trend, ADX<20 = range)
3. DAILY RSI(14): Entry timing (pullback to 40-60 in trend, extremes in range)
4. DAILY BB WIDTH(20): Regime detection (width percentile <30 = squeeze breakout)
5. ATR(14) trailing stop: 3x ATR for risk management
6. Position sizing: 0.30 discrete (conservative for daily moves)

ENTRY LOGIC:
- BULL regime (price > weekly HMA): Long on RSI pullback to 35-50 + ADX>20
- BEAR regime (price < weekly HMA): Short on RSI rally to 50-65 + ADX>20
- RANGE regime (ADX<20): Mean-revert at RSI<30 or RSI>70 + BB squeeze

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_weekly_hma_adx_rsi_bb_ensemble_atr_v1"
timeframe = "1d"
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
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and DM
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    adx[period*2:] = adx_raw[period*2:]  # ADX needs 2x period to stabilize
    return adx

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / (sma + 1e-10)  # Normalized width
    return upper, lower, sma, width

def calculate_percentile_rank(values, lookback=100):
    """Calculate percentile rank of current value vs lookback period."""
    n = len(values)
    pr = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = values[i-lookback:i]
        current = values[i]
        if len(window) > 0:
            pr[i] = np.sum(window < current) / len(window) * 100
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Weekly HMA trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # Daily ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        # 20-25 = neutral
        
        # BB Width regime (squeeze = potential breakout)
        bb_squeeze = bb_width_pr[i] < 30  # Width in bottom 30% of last 100 days
        
        # === SIGNAL VOTES (Ensemble: need 2 of 3) ===
        long_votes = 0
        short_votes = 0
        
        # Vote 1: Trend alignment
        if bull_trend_1w:
            long_votes += 1
        elif bear_trend_1w:
            short_votes += 1
        
        # Vote 2: RSI entry timing
        # In bull trend: look for pullback (RSI 35-55)
        # In bear trend: look for rally (RSI 45-65)
        # In range: look for extremes (RSI <30 or >70)
        if bull_trend_1w and 35 <= rsi[i] <= 55:
            long_votes += 1
        elif bear_trend_1w and 45 <= rsi[i] <= 65:
            short_votes += 1
        elif weak_trend:
            if rsi[i] < 30:
                long_votes += 1
            elif rsi[i] > 70:
                short_votes += 1
        
        # Vote 3: ADX confirmation or BB squeeze
        if strong_trend:
            if bull_trend_1w:
                long_votes += 1
            elif bear_trend_1w:
                short_votes += 1
        elif bb_squeeze:
            # Squeeze breakout - follow trend direction
            if bull_trend_1w:
                long_votes += 1
            elif bear_trend_1w:
                short_votes += 1
        
        # === GENERATE SIGNAL (need 2 of 3 votes) ===
        new_signal = 0.0
        
        if long_votes >= 2:
            new_signal = SIZE
        elif short_votes >= 2:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w and adx[i] > 25:
                # Long position but weekly trend flipped bearish with strong ADX
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w and adx[i] > 25:
                # Short position but weekly trend flipped bullish with strong ADX
                new_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 75:
                # Long position, RSI overbought - take profit
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 25:
                # Short position, RSI oversold - take profit
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals