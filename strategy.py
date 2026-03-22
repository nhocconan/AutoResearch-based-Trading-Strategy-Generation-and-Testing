#!/usr/bin/env python3
"""
Experiment #073: 1d Primary + 1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Single-regime strategies fail because crypto alternates between trending
and ranging markets. This strategy adapts to market regime:

REGIME 1 - TRENDING (ADX > 25):
- Follow 1w HMA(21) slope direction
- Enter on 1d HMA(8/21) crossover with trend
- RSI(14) confirms momentum (50-70 for long, 30-50 for short)
- Exit on trend reversal or 2.5 ATR stoploss

REGIME 2 - RANGING (ADX <= 25):
- Mean revert at Bollinger Band extremes
- Long when price < BB_lower + RSI < 35 + 1w HMA not strongly bearish
- Short when price > BB_upper + RSI > 65 + 1w HMA not strongly bullish
- Exit at BB middle or opposite band

Why this should work:
- 1d timeframe naturally limits to 30-60 trades/year
- Regime adaptation prevents trend strategies in chop and vice versa
- 1w HTF filter prevents counter-trend trades in strong weekly trends
- Discrete position sizing (0.25/0.30) minimizes fee churn
- ATR stoploss protects against black swan events

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.30 for strong signals
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_adx_bb_1w_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope over lookback period."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, 20, 2.0)
    
    # HMA for trend following entries
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # Strong bullish: slope > 2%
        # Strong bearish: slope < -2%
        # Neutral: -2% to 2%
        week_bullish_strong = hma_1w_slope_aligned[i] > 2.0
        week_bearish_strong = hma_1w_slope_aligned[i] < -2.0
        week_neutral = not week_bullish_strong and not week_bearish_strong
        
        week_bullish = hma_1w_slope_aligned[i] > 0
        week_bearish = hma_1w_slope_aligned[i] < 0
        
        # === REGIME DETECTION ===
        # ADX > 25 = trending regime (follow trend)
        # ADX <= 25 = ranging regime (mean revert)
        trending_regime = adx_14[i] > 25
        ranging_regime = adx_14[i] <= 25
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if trending_regime and (week_bullish_strong or week_bearish_strong):
            current_size = STRONG_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TRENDING REGIME: Follow 1w trend on 1d HMA crossover
        if trending_regime:
            # LONG: 1w bullish + 1d HMA bullish cross + RSI confirmation
            if week_bullish:
                hma_bull_cross = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
                hma_bull_aligned = hma_8[i] > hma_21[i]
                rsi_ok = 45 < rsi_14[i] < 70
                
                if hma_bull_cross and rsi_ok:
                    new_signal = current_size
                elif hma_bull_aligned and rsi_ok and bars_since_last_trade > 30:
                    new_signal = BASE_SIZE
            
            # SHORT: 1w bearish + 1d HMA bearish cross + RSI confirmation
            if week_bearish:
                hma_bear_cross = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
                hma_bear_aligned = hma_8[i] < hma_21[i]
                rsi_ok = 30 < rsi_14[i] < 55
                
                if hma_bear_cross and rsi_ok:
                    new_signal = -current_size
                elif hma_bear_aligned and rsi_ok and bars_since_last_trade > 30:
                    new_signal = -BASE_SIZE
        
        # RANGING REGIME: Mean revert at BB extremes
        elif ranging_regime:
            price_near_lower = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower
            price_near_upper = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper
            
            # LONG: Price at BB lower + RSI oversold + 1w not strongly bearish
            if price_near_lower and rsi_14[i] < 38 and not week_bearish_strong:
                new_signal = BASE_SIZE
            
            # SHORT: Price at BB upper + RSI overbought + 1w not strongly bullish
            if price_near_upper and rsi_14[i] > 62 and not week_bullish_strong:
                new_signal = -BASE_SIZE
            
            # BB squeeze breakout (vol expansion after compression)
            bb_width = (bb_upper[i] - bb_lower[i]) / bb_middle[i] if bb_middle[i] != 0 else 0
            bb_width_prev = (bb_upper[i-5] - bb_lower[i-5]) / bb_middle[i-5] if bb_middle[i-5] != 0 else 0
            
            if bb_width > bb_width_prev * 1.2:  # Width expanding 20%
                if week_bullish and hma_8[i] > hma_21[i] and rsi_14[i] > 50:
                    new_signal = BASE_SIZE
                elif week_bearish and hma_8[i] < hma_21[i] and rsi_14[i] < 50:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~60 days on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if week_bullish and hma_8[i] > hma_21[i] and rsi_14[i] > 48:
                new_signal = BASE_SIZE * 0.6
            elif week_bearish and hma_8[i] < hma_21[i] and rsi_14[i] < 52:
                new_signal = -BASE_SIZE * 0.6
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w trend reverses strongly bearish
            if position_side > 0 and week_bearish_strong:
                trend_reversal = True
            # Exit short if 1w trend reverses strongly bullish
            if position_side < 0 and week_bullish_strong:
                trend_reversal = True
            
            # Exit in ranging regime if price reaches BB middle (take profit)
            if ranging_regime:
                if position_side > 0 and close[i] >= bb_middle[i]:
                    trend_reversal = True
                if position_side < 0 and close[i] <= bb_middle[i]:
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