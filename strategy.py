#!/usr/bin/env python3
"""
Experiment #017: 1d HMA Trend + RSI Pullback + 1w HTF Filter with ATR Stoploss

Hypothesis: Daily timeframe reduces noise and whipsaw compared to lower TFs.
Combining 1d HMA trend with RSI pullback entries and 1w HTF bias should:
1. Reduce trade frequency to 20-50/year (fee-efficient)
2. Capture major trend moves while avoiding counter-trend trades
3. Use 1w HMA as ultimate trend filter (only trade with weekly bias)
4. RSI pullback entries (RSI<40 in uptrend, RSI>60 in downtrend) for better entry timing
5. ATR trailing stoploss (2.5*ATR) to protect gains and limit losses

Why 1d should work better than failed 4h/12h strategies:
- Less noise, fewer false signals
- Lower fee drag (20-50 trades/year vs 100+ on 4h)
- Better suited for BTC/ETH which have strong multi-day trends
- 2022 crash and 2025 bear market better captured on daily

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_filter_atr_stop_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
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
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    di_sum = plus_di + minus_di
    di_sum = np.where(di_sum == 0, 1e-10, di_sum)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        # === HTF TREND BIAS (1w) ===
        # Weekly HMA determines overall bias
        htf_bullish = close[i] > hma_1w_21_aligned[i]
        htf_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D HMA TREND ===
        # HMA21 > HMA50 = bullish structure
        hma_bullish = (hma_1d_21[i] > hma_1d_50[i]) and (close[i] > hma_1d_21[i])
        hma_bearish = (hma_1d_21[i] < hma_1d_50[i]) and (close[i] < hma_1d_21[i])
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20  # Trend present
        adx_weak = adx_14[i] < 15  # No trend
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-45 in uptrend
        # Short: RSI rallied to 55-65 in downtrend
        rsi_long_pullback = (rsi_14[i] >= 35) and (rsi_14[i] <= 45)
        rsi_short_pullback = (rsi_14[i] >= 55) and (rsi_14[i] <= 65)
        
        # === EXTREME RSI REVERSAL ===
        # Long: RSI < 30 (oversold) in bullish HTF
        # Short: RSI > 70 (overbought) in bearish HTF
        rsi_extreme_long = rsi_14[i] < 30
        rsi_extreme_short = rsi_14[i] > 70
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: HTF bullish + HMA bullish + (RSI pullback OR RSI extreme)
        if htf_bullish and hma_bullish:
            if rsi_long_pullback and adx_strong:
                new_signal = current_size
            elif rsi_extreme_long:
                # Extreme oversold entry (weaker ADX requirement)
                new_signal = current_size * 0.8
        
        # SHORT ENTRY: HTF bearish + HMA bearish + (RSI pullback OR RSI extreme)
        elif htf_bearish and hma_bearish:
            if rsi_short_pullback and adx_strong:
                new_signal = -current_size
            elif rsi_extreme_short:
                # Extreme overbought entry (weaker ADX requirement)
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~60 days on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if htf_bullish and hma_bullish and rsi_14[i] < 50:
                new_signal = current_size * 0.6
            elif htf_bearish and hma_bearish and rsi_14[i] > 50:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                htf_reversal = True
            if position_side < 0 and htf_bullish:
                htf_reversal = True
        
        # === ADX WEAK EXIT ===
        adx_exit = False
        if in_position and position_side != 0 and adx_weak:
            # Exit if trend strength disappears
            adx_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or htf_reversal or adx_exit:
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