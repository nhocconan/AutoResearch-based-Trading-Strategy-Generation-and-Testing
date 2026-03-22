#!/usr/bin/env python3
"""
Experiment #205: 15m EMA Pullback + 4h HMA Trend + 1h RSI Momentum + BB Regime + ATR Stop

Hypothesis: 15m timeframe captures intraday swings while 4h HMA provides stable trend bias.
EMA pullback entries (price retraces to EMA21 in trend direction) combined with 1h RSI 
momentum confirmation should generate more trades than pure breakout strategies while 
maintaining quality. Bollinger Band regime filter avoids trading during extreme volatility.

Why 15m might work:
- 15m = 96 bars/day, captures intraday mean reversion + trend continuation
- EMA pullback = buy dips in uptrend, sell rallies in downtrend (proven edge)
- 4h HMA filter = prevents counter-trend trades (major failure mode in crypto)
- 1h RSI momentum = confirms pullback is ending, not continuing
- BB regime = avoid trading when bands too wide (volatility crush risk)
- Conservative sizing (0.25) + ATR stop = controls drawdown

Learning from failures:
- #193 (15m pullback): Sharpe=-3.489 - likely too strict entry or wrong HTF
- #199 (15m KAMA): Sharpe=-4.763 - KAMA alone doesn't work on 15m
- Pure trend following fails on 15m (too much noise)
- Pure mean reversion fails without trend filter
- Need BOTH trend filter AND momentum confirmation for 15m

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_ema_pullback_4h_hma_1h_rsi_bb_atr_v1"
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return upper.values, lower.values, bandwidth.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger(close, 20, 2.0)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    profit_target_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND STRUCTURE ===
        # EMA21 > EMA50 = bullish structure
        # EMA21 < EMA50 = bearish structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === BOLLINGER BAND REGIME ===
        # BB bandwidth < 0.10 = squeeze (low vol, breakout likely)
        # BB bandwidth > 0.25 = extreme vol (avoid trading)
        bb_squeeze = bb_bandwidth[i] < 0.10
        bb_extreme = bb_bandwidth[i] > 0.25
        
        # === RSI MOMENTUM ===
        # RSI < 40 = oversold (long opportunity in uptrend)
        # RSI > 60 = overbought (short opportunity in downtrend)
        # Use flexible thresholds to ensure trade count
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # === PRICE VS EMA PULLBACK ===
        # Price near EMA21 = pullback entry zone
        # Price > EMA21 * 1.02 = extended (don't chase)
        # Price < EMA21 * 0.98 = extended (don't chase)
        price_vs_ema_long = close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.97
        price_vs_ema_short = close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.03
        
        # === ADX TREND STRENGTH ===
        # ADX > 15 = some trend (lower threshold for 15m to ensure trades)
        trend_strength = adx[i] > 15
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + EMA bullish + RSI oversold + price at EMA + not extreme vol
        # Flexible conditions to ensure enough trades
        if bull_trend_4h and ema_bullish and not bb_extreme:
            if rsi_oversold and price_vs_ema_long:
                new_signal = SIZE_BASE
            elif rsi[i] < 50 and close[i] < ema_21[i] * 0.99:
                # Deeper pullback entry
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + EMA bearish + RSI overbought + price at EMA + not extreme vol
        if bear_trend_4h and ema_bearish and not bb_extreme:
            if rsi_overbought and price_vs_ema_short:
                new_signal = -SIZE_BASE
            elif rsi[i] > 50 and close[i] > ema_21[i] * 1.01:
                # Deeper rally entry
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT LOGIC ===
        # Reduce position at 2R profit, trail stop
        if in_position and position_side != 0 and not profit_target_hit:
            if position_side > 0:
                profit_target = entry_price + 2.0 * atr[i]
                if close[i] > profit_target:
                    new_signal = SIZE_HALF  # Reduce to half position
                    profit_target_hit = True
            if position_side < 0:
                profit_target = entry_price - 2.0 * atr[i]
                if close[i] < profit_target:
                    new_signal = -SIZE_HALF  # Reduce to half position
                    profit_target_hit = True
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
                profit_target_hit = False
            elif abs(new_signal) < abs(signals[i-1]) if i > 0 else False:
                # Reducing position (take profit)
                profit_target_hit = True
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
                profit_target_hit = False
        
        signals[i] = new_signal
    
    return signals