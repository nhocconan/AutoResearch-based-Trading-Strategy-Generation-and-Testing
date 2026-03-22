#!/usr/bin/env python3
"""
Experiment #004: 4h Volatility Mean Reversion with 12h Trend Filter

Hypothesis: Previous Choppiness Index regime strategies failed (3x negative Sharpe).
New approach: Volatility spike mean reversion + ADX trend filter + 12h HMA bias.

Why this should work:
1. Vol spike reversion (ATR(7)/ATR(30) > 2.0) captures panic reversals - proven edge
2. Bollinger Band extremes (price < lower BB) confirm oversold/overbought
3. ADX > 20 filters out dead chop (unlike Choppiness which failed)
4. 12h HMA provides major trend bias without over-filtering
5. 4h timeframe = 20-50 trades/year target (fee-efficient)
6. Discrete position sizing (0.25/0.30) minimizes churn costs

Key differences from failed experiments:
- NO Choppiness Index (failed 3 consecutive times)
- Uses ADX for trend strength (more reliable)
- Vol spike detection (ATR ratio) for entry timing
- Simpler logic = more robust across BTC/ETH/SOL

Timeframe: 4h (REQUIRED for exp#004)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_mean_reversion_12h_trend_v1"
timeframe = "4h"
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
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate DM+ and DM-
    high_diff = high_s.diff()
    low_diff = -low_s.diff()
    
    dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    tr_smooth = np.where(tr_smooth == 0, 1e-10, tr_smooth)
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # Calculate DX and ADX
    di_sum = di_plus + di_minus
    di_sum = np.where(di_sum == 0, 1e-10, di_sum)
    dx = 100 * np.abs(di_plus - di_minus) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_50 = calculate_hma(df_12h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_50_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    adx_14 = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    entry_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_50_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        
        # === 12H TREND BIAS ===
        # Simple: price above 12h HMA(21) = bullish bias
        trend_bullish = close[i] > hma_12h_21_aligned[i]
        trend_bearish = close[i] < hma_12h_21_aligned[i]
        
        # Stronger confirmation: HMA(21) > HMA(50)
        hma_confirmed_bullish = hma_12h_21_aligned[i] > hma_12h_50_aligned[i]
        hma_confirmed_bearish = hma_12h_21_aligned[i] < hma_12h_50_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7) / ATR(30) > 1.8 = volatility spike (panic/extreme)
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 1.0
        vol_spike = atr_ratio > 1.8
        
        # === ADX TREND STRENGTH ===
        # ADX > 20 = trending (avoid dead chop)
        # ADX < 25 = not strongly trending (good for mean reversion)
        adx_trending = adx_14[i] > 20.0
        adx_not_strong = adx_14[i] < 35.0
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
        near_lower_bb = bb_position < 0.15  # Price in bottom 15% of BB
        near_upper_bb = bb_position > 0.85  # Price in top 15% of BB
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_median = np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else atr_14[i]
        atr_ratio_norm = atr_14[i] / atr_median if atr_median > 0 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio_norm, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (Volatility Mean Reversion) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: Vol spike + near lower BB + RSI oversold + 12h trend not bearish
        if vol_spike and near_lower_bb and rsi_oversold:
            if not trend_bearish or hma_confirmed_bullish:
                new_signal = current_size
        
        # SHORT: Vol spike + near upper BB + RSI overbought + 12h trend not bullish
        elif vol_spike and near_upper_bb and rsi_overbought:
            if not trend_bullish or hma_confirmed_bearish:
                new_signal = -current_size
        
        # Secondary entry: BB extreme without vol spike (less common)
        elif near_lower_bb and rsi_oversold and adx_not_strong:
            if hma_confirmed_bullish:
                new_signal = current_size * 0.7
        
        elif near_upper_bb and rsi_overbought and adx_not_strong:
            if hma_confirmed_bearish:
                new_signal = -current_size * 0.7
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~6-7 days on 4h), force entry with weaker conditions
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if hma_confirmed_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.5
            elif hma_confirmed_bearish and rsi_14[i] > 55:
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
            
            elif position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend turns bearish strongly
            if position_side > 0 and hma_confirmed_bearish and adx_trending:
                trend_reversal = True
            # Exit short if 12h trend turns bullish strongly
            if position_side < 0 and hma_confirmed_bullish and adx_trending:
                trend_reversal = True
        
        # === TIME-BASED EXIT ===
        # Exit after 60 bars (~10 days) if no profit
        bars_in_trade = i - entry_bar if entry_bar > 0 else 0
        time_exit = False
        if in_position and bars_in_trade > 60:
            # Check if we're at small loss or profit
            if position_side > 0 and close[i] > entry_price * 0.98:
                time_exit = True
            elif position_side < 0 and close[i] < entry_price * 1.02:
                time_exit = True
        
        # Apply stoploss or trend reversal or time exit
        if stoploss_triggered or trend_reversal or time_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        signal_changed = (signals[i-1] != 0 and new_signal == 0) or (signals[i-1] == 0 and new_signal != 0) or (np.sign(signals[i-1]) != np.sign(new_signal) and new_signal != 0)
        
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position and (stoploss_triggered or trend_reversal or time_exit or signal_changed):
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = -50
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals