#!/usr/bin/env python3
"""
Experiment #426: 1d HMA Trend + RSI Pullback + Weekly HTF Confirmation

Hypothesis: Daily timeframe needs SIMPLER entry logic to generate sufficient trades.
Previous 12h/1d strategies failed due to overly strict regime filters (ADX>25 rarely
triggers on daily bars). This strategy uses:

1. HMA(21) on 1d for primary trend direction (smoother than EMA, less lag)
2. RSI(7) for faster entry timing on daily bars (RSI14 too slow for 1d)
3. ADX(14) with LOWER threshold (18 instead of 25) for 1d regime detection
4. Weekly HMA(21) via mtf_data for HTF trend confirmation
5. ATR(14) trailing stop at 2.5x for risk management
6. Bollinger Band squeeze detection for volatility expansion entries

ENTRY LOGIC (simplified to ensure 10+ trades/year):
- Long: price > HMA21 + RSI7 < 45 (pullback in uptrend) + ADX > 18
- Short: price < HMA21 + RSI7 > 55 (rally in downtrend) + ADX > 18
- Weekly HMA filter: only long if price > weekly HMA, only short if price < weekly HMA

Why this should work on 1d:
- Fewer but higher-quality signals (20-40 trades/year vs 100+ on 4h)
- Less fee drag from reduced churn
- Weekly HTF alignment prevents counter-trend trades
- RSI7 faster than RSI14, catches more pullback opportunities
- ADX threshold 18 (not 25) ensures enough trending periods trigger

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi7_weekly_hma_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di = 100 * plus_dm_s[i] / tr_s[i]
            minus_di = 100 * minus_dm_s[i] / tr_s[i]
            di_sum = plus_di + minus_di
            if di_sum > 1e-10:
                dx = 100 * np.abs(plus_di - minus_di) / di_sum
            else:
                dx = 0
        else:
            dx = 0
        
        if i == period:
            adx[i] = dx
        else:
            adx[i] = ((adx[i-1] * (period - 1)) + dx) / period
    
    return adx

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index with faster period for 1d."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values

def calculate_bb_width(upper, lower, middle):
    """Calculate Bollinger Band Width (volatility measure)."""
    width = (upper - lower) / middle
    return width

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
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 7)  # Faster RSI for daily
    hma_21 = calculate_hma(close, 21)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_mid)
    
    # Calculate BB width percentile for squeeze detection
    bb_width_pct = pd.Series(bb_width).rolling(window=60, min_periods=60).rank(pct=True).values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0 or np.isnan(atr[i-1]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d HMA) ===
        bull_trend_1d = close[i] > hma_21[i]
        bear_trend_1d = close[i] < hma_21[i]
        
        # === WEEKLY HTF TREND CONFIRMATION ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (ADX with lower threshold for 1d) ===
        trending_market = adx[i] > 18  # Lower threshold for daily
        
        # === RSI PULLBACK SIGNALS ===
        rsi_pullback_long = rsi[i] < 45  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 55  # Rally in downtrend
        
        # === BOLLINGER SQUEEZE (volatility expansion setup) ===
        bb_squeeze = bb_width_pct[i] < 0.30  # Bottom 30% of BB width
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d uptrend + weekly uptrend + RSI pullback + trending
        if bull_trend_1d and bull_trend_1w and rsi_pullback_long and trending_market:
            new_signal = SIZE
        
        # SHORT ENTRY: 1d downtrend + weekly downtrend + RSI rally + trending
        elif bear_trend_1d and bear_trend_1w and rsi_pullback_short and trending_market:
            new_signal = -SIZE
        
        # BOLLINGER SQUEEZE BREAKOUT (alternative entry when ADX low)
        elif bb_squeeze and not trending_market:
            if bull_trend_1d and bull_trend_1w and rsi_pullback_long:
                new_signal = SIZE
            elif bear_trend_1d and bear_trend_1w and rsi_pullback_short:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            # Long position exits if 1d trend reverses
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            # Short position exits if 1d trend reverses
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === WEEKLY TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            # Long position exits if weekly trend reverses
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            # Short position exits if weekly trend reverses
            if position_side < 0 and bull_trend_1w:
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